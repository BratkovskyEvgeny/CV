import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def detect_objects(our_image):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    col1, col2 = st.columns(2)

    col1.subheader("Исходное изображение")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(our_image)
    col1.pyplot(use_column_width=False)

    #инициализация алгоритма "YOLO"
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0,255,size=(len(classes), 3))   


    #загрузка изображения
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    height,width,channels = img.shape


    #конвертация в blob 
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False)   

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes =[]

    #для отображения информации после детекции
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)  
            confidence = scores[class_id] 
            if confidence > 0.5:   
                
                #получение координат объекта  
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)  
                h = int(detection[3] * height) 

                # координаты рамки (прямоугольника)
                x = int(center_x - w /2)   
                y = int(center_y - h/2)   

                
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    score_threshold = st.sidebar.slider("Confidence threshold", 0.00,1.00,0.5,0.01)
    nms_threshold = st.sidebar.slider("NMS threshold", 0.00, 1.00, 0.4, 0.01)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences,score_threshold,nms_threshold)      
    print(indexes)

    font = cv2.FONT_HERSHEY_SIMPLEX
    items = []
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            #для получения наименований объектов
            label = str.upper((classes[class_ids[i]]))   
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,3)     
            items.append(label)


    st.text("")
    col2.subheader("Обнаруженные объекты")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(img)
    col2.pyplot(use_column_width=True)

    if len(indexes)>1:
        st.success("Детектирован(о/ы) {} объект(а/ов) класса(ов) - {}".format(len(indexes),[item for item in set(items)]))
    else:
        st.success("Детектирован {} объект класса - {}".format(len(indexes),[item for item in set(items)]))


def object_main():
    st.sidebar.image('logo.png', width=300)
    #st.markdown("<h1 style='text-align: center; color: black;'>Веб-приложение для демонстрации развёртывания модели компьютерного зрения</h1>", unsafe_allow_html = True)
    #st.sidebar.title("Веб-приложение для демонстрации модели компьютерного зрения (проект для базового потока Deep Learning School)")
    st.sidebar.info("Веб-приложение для демонстрации алгоритма детекции (обнаружения) объектов (проект для базового потока Deep Learning School). Создано Братковским Евгением Викторовичем (User ID (Stepik.org): 528271325).")
    st.title("Обнаружение объектов")
    st.write("<h6 style='text-align: center;'>YOLO или You Only Look Once — это популярная архитектура CNN, которая используется для распознавания объектов на изображении. В проекте использована YOLOv3 — усовершенствованная версия архитектуры YOLO. Основная особенность YOLOv3 состоит в том, что на выходе есть три слоя, каждый из которых расчитан на обнаружение объектов разного размера. Наибольшим преимуществом YOLO над другими архитектурами является скорость. Модели семейства YOLO исключительно быстры и намного превосходят R-CNN (Region-Based Convolutional Neural Network) и другие модели.</h6>", unsafe_allow_html = True)
    #st.image('yolo_1.png', width=350)
    choice = st.radio("", ("Пример для демонстрации", "Выбрать изображение из коллекции"))
    st.write()

    if choice == "Выбрать изображение из коллекции":
        st.set_option('deprecation.showfileUploaderEncoding', False)
        image_file = st.file_uploader("Выбор изображений", type=['jpg','png','jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)  
            detect_objects(our_image)

    elif choice == "Пример для демонстрации":
        our_image = Image.open("my_jawa634.jpg")
        detect_objects(our_image)
    #st.sidebar.info("Веб-приложение для демонстрации модели компьютерного зрения (проект для базового потока Deep Learning School)")

if __name__ == '__main__':
    object_main()
