# API de Clasificación de Tumores

Este proyecto implementa un modelo de machine learning desplegado como una API REST utilizando FastAPI.

## Funcionalidades

* Clasificación de tumores (benigno / maligno)
* Endpoint principal: /diagnosticar
* Monitoreo con logging
* Métricas del sistema (/metricas)
* Health check (/salud)

## Tecnologías utilizadas

* FastAPI
* Scikit-learn
* Uvicorn
* Ngrok
* Python

## Cómo ejecutar

pip install -r requirements.txt
uvicorn app:app --reload

Luego abrir en el navegador:
http://localhost:8000/docs
https://interbelligerent-nenita-monorhinous.ngrok-free.dev/docs

## Ejemplo de uso

POST /diagnosticar

Request:
{
"mean_radius": 14.5,
"mean_texture": 20.1,
"mean_perimeter": 92.3,
"mean_area": 650
}
Nota: el modelo requiere 30 características como entrada.
Response:
{
"diagnostico": "maligno",
"confianza_porcentaje": 99.78
}

## Estructura del proyecto

* app.py
* modelo_cancer.pkl
* scaler.pkl
* features.pkl
* requirements.txt

## Autoras

* Jesica Natalia Chaparro Perez
* Maria Alejandra Gamboa
* Leyde Katerine Cortes
* Maria Camila Forero
* Adriana Lucia Cristancho
