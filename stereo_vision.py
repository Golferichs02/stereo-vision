import argparse
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Global variables to manage coordinates and selection state
coords = []
select_left = True  # Start by selecting from the left image

def user_interaction():
    """
    Commando parser que permite la interaccion con la terminal, donde se
    obtendra la ruta de las 2 imagenes estereo, ruta de los datos de 
    calibracion y el numero entero para realizar un redimensionamiento.
    """
    parser = argparse.ArgumentParser(description='3D Reconstruction from Stereo Images')
    parser.add_argument('-l', '--left_img',
                        type=str,
                        required=True,
                        help="Path to the left rectified image")
    parser.add_argument('-r', '--right_img',
                        type=str,
                        required=True,
                        help="Path to the right rectified image")
    parser.add_argument('-c', '--calib_file',
                        type=str,
                        required=True,
                        help="Path to the calibration file in txt format")
    parser.add_argument('-s', '--resize',
                        type=int,
                        required=True,
                        help="Resize (integral number to get %)")
    
    return parser.parse_args()

def read_calibration(file_path:str)->dict:
    """
    Reads camera calibration parameters from a plain text file.

    Args:
        file_path: ruta donde se encuentra el panel de calibracion original

    Returns:
        dict: Diccionario de los parametros leidos
    """
    calibration_params = {}
    with open(file_path, 'r') as file:
        for line in file:
            if ":" in line:
                key, value = line.split(":")
                # Remover comillas, espacios adicionales y comas
                key = key.strip().replace('"', '')
                value = value.strip().replace(',', '').replace('"', '')
                # Convertir el valor a float si es posible
                try:
                    value = float(value)
                except ValueError:
                    pass
                calibration_params[key] = value
    return calibration_params

def resize_image_and_calibrate(img_path:str, calibration:dict, resize_percent:int)->{cv2,dict}:
    """
    Resizes an image and adjusts calibration parameters.

    Args:
        img_path: Direccion de la imagen.
        calibration: Formato de calibracion original
        resize_percent: Numero entero que representa el % del resize a aplicar

    Returns:
        tuple: Tuple containing the resized image and adjusted calibration parameters.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Calcular las nuevas dimensiones
    width = int(img.shape[1] * resize_percent / 100)
    height = int(img.shape[0] * resize_percent / 100)
    dim = (width, height)
    # Redimensionar la imagen
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # Ajustar los parámetros de calibración

    resized_calibration = {
        "baseline": float(calibration["baseline"]),
        "rectified_fx": float(calibration["rectified_fx"]) * resize_percent / 100,
        "rectified_fy": float(calibration["rectified_fy"]) * resize_percent / 100,
        "rectified_cx": float(calibration["rectified_cx"]) * resize_percent / 100,
        "rectified_cy": float(calibration["rectified_cy"]) * resize_percent / 100,
        "rectified_width": width,
        "rectified_height": height
    }

    return resized_img, resized_calibration

def zoom(event:int)->None:
    """
    Ajusta los limites x e y de la imagen. 

    Parámetros:
    event : MouseEvent
        Detecta el movimiento ejercido en la rueda del mouse
    """
    ax = event.inaxes
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_data, y_data = event.xdata, event.ydata

    if event.button == 'up':
        scale_factor = 1 / 1.5
    elif event.button == 'down':
        scale_factor = 1.5
    ax.set_xlim([x_data - (x_data - x_min) * scale_factor,
                 x_data + (x_max - x_data) * scale_factor])
    ax.set_ylim([y_data - (y_data - y_min) * scale_factor,
                 y_data + (y_max - y_data) * scale_factor])
    ax.figure.canvas.draw()

def get_pixel_coords(event:int)->None:
    """
    Captura y almacena coordenadas de píxeles a partir de eventos de clic de ratón en una imagen.

    Parámetros:
    event : MouseEvent
        El evento de ratón que incluye información sobre la ubicación del clic y los ejes.
    """
    global select_left, coords
    if event.inaxes is None or event.button != 1:  # Ignore if click is outside axes or not left click
        return

    x, y = int(event.xdata), int(event.ydata)
    if select_left:
        coords.append((x, y))  # Store left image coordinates
        print(f"Left Image - Coordenadas capturadas: ({x}, {y})")
        event.inaxes.set_title('Left Image (done)')
        select_left = False  # Toggle to right image
    else:
        coords.append((x, y))  # Store right image coordinates
        print(f"Right Image - Coordenadas capturadas: ({x}, {y})")
        event.inaxes.set_title('Right Image (done)')
        select_left = True  # Toggle back to left image

def display_images_and_capture_points(img_left:plt, img_right:plt)->None:
    """
    Muestra dos imágenes lado a lado y configura manejadores de eventos para 
    zoom y captura de coordenadas.

    Parámetros:
    img_left : Matriz imagen izquirda
    img_right : Matriz imagen derecha
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(img_left, cmap='gray')
    axs[0].set_title('Left Image (start here)')
    axs[0].axis('off')

    axs[1].imshow(img_right, cmap='gray')
    axs[1].set_title('Right Image')
    axs[1].axis('off')

    fig.canvas.mpl_connect('scroll_event', zoom)
    fig.canvas.mpl_connect('button_press_event', get_pixel_coords)

    plt.show()

def compute_3d_coordinates(uL:int, vL:int, uR:int, calibration:dict)->tuple:
    """
    Calcula las coordenadas 3D y sus errores.

    Args:
        uL : Coordenada X en la imagen izquierda.
        vL : Coordenada Y en la imagen izquierda.
        uR : Coordenada X en la imagen derecha.
        calibration : Parámetros de calibración.

    Returns:
        tuple: Coordenadas (X, Y, Z) y sus errores.
    """
    cx = float(calibration["rectified_cx"])
    cy = float(calibration["rectified_cy"])
    f = float(calibration["rectified_fx"])
    B = abs(float(calibration["baseline"]))

    ucL = (uL - cx) 
    ucR = (uR - cx) 
    vcL = (vL - cy) 
    disparity = ucL - ucR

    if disparity == 0:
        Z = float('inf')
    else:
        Z = (f * B) / disparity

    X = ucL * Z / f
    Y = vcL * Z / f

    delta_X, delta_Y, delta_Z = compute_3d_errors(Z, f, B)

    return (X, Y, Z), (delta_X, delta_Y, delta_Z)

def compute_3d_errors(Z:float, f:float, B:float, disparity_error=1.0)->tuple:
    """
    Calcula los errores en las coordenadas 3D (X, Y, Z).

    Args:
        Z : Coordenada Z calculada.
        f : Longitud focal.
        B : Distancia entre cámaras.
        disparity_error: Error de píxel en la disparidad .

    Returns:
        Tupla: Errores en X, Y y Z.
    """
    delta_X = disparity_error * (Z / f)
    delta_Y = disparity_error * (Z / f)
    delta_Z = (Z ** 2) / (f * B) * disparity_error

    return delta_X, delta_Y, delta_Z

def calculate_coords(calibration_resize:dict)->{list,list}:
    # Calcular las coordenadas 3D    
    for i in range(0, len(coords), 2):
        uL, vL = coords[i]
        uR, _ = coords[i + 1]
        (X, Y, Z), (delta_X, delta_Y, delta_Z) = compute_3d_coordinates(uL, vL, uR, calibration_resize)
        coordinates_3d.append((X, Y, Z))
        errors.append((delta_X, delta_Y, delta_Z))
    return coordinates_3d,errors

def plot_3d_coordinates(coordinates_3d:list)->None:
    """
    Imprime la grafica 3D con sus respectivas coordenadas en el plano

    Args:
        coordinates_3d :Lista de las coordenadas seleccionadas para el plano 3D
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x, y, z in coordinates_3d:
        ax.scatter(x, z, -y)
    ax.set_xlabel('X')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    plt.show()

def print_coordinates_and_errors(coordinates_3d:list, errors:list)->None:
    """
    Imprime las coordenadas 3D y sus errores.

    Args:
        coordinates_3d : Lista de tuplas (X, Y, Z) que representan las coordenadas 3D.
        errors: Lista de tuplas (delta_X, delta_Y, delta_Z) que representan los errores asociados
        a sus respectivas coordenadas.
    """
    for i, ((X, Y, Z), (delta_X, delta_Y, delta_Z)) in enumerate(zip(coordinates_3d, errors)):
        print(f"Punto {i+1}: Coordenadas 3D: X = {X:.2f} ± {delta_X:.2f}, Y = {Y:.2f} ± {delta_Y:.2f}, Z = {Z:.2f} ± {delta_Z:.2f}")
        

    
coordinates_3d = []
errors = []      
def pipeline():
    """
    Ejecuta el proceso completo de reconstrucción 3D desde 
    la interacción del usuario hasta la visualización de resultados.
    """
    args = user_interaction() #permitir interaccion con terminal
    calibration = read_calibration(args.calib_file) #leer parametros de calibracion
    # Redimensionar las imágenes y ajustar la calibración
    img_left, calibration_resize = resize_image_and_calibrate(args.left_img, calibration, args.resize)
    img_right, _ = resize_image_and_calibrate(args.right_img, calibration, args.resize)
    display_images_and_capture_points(img_left, img_right) #mostrar imagenes y permitir interaccion con clics
    coordinates_3d, errors=calculate_coords(calibration_resize)# Calcular coordenadas y errores 
    print_coordinates_and_errors(coordinates_3d, errors)# Imprimir todas las coordenadas y errores
    plot_3d_coordinates(coordinates_3d) #Imprimir el plano 3d

if __name__ == "__main__":
    pipeline()