
import logging
from collections import defaultdict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_cancer.log")
    ]
)
logger = logging.getLogger("api_cancer")

metricas = defaultdict(int)
metricas["inicio_servicio"] = datetime.now().isoformat()

import joblib
import numpy as np

try:
    modelo = joblib.load("modelo_cancer.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    logger.info("Modelo cargado correctamente")
except Exception as e:
    logger.critical(f"Error al cargar el modelo: {e}")
    raise

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="API de Diagnóstico de Tumores",
    description="Clasifica tumores como malignos o benignos usando Gradient Boosting.",
    version="1.0.0"
)

class DatosTumor(BaseModel):
    mean_radius: float = Field(..., example=14.5)
    mean_texture: float = Field(..., example=20.1)
    mean_perimeter: float = Field(..., example=92.3)
    mean_area: float = Field(..., example=650.0)
    mean_smoothness: float = Field(..., example=0.096)
    mean_compactness: float = Field(..., example=0.104)
    mean_concavity: float = Field(..., example=0.089)
    mean_concave_points: float = Field(..., example=0.050)
    mean_symmetry: float = Field(..., example=0.181)
    mean_fractal_dimension: float = Field(..., example=0.063)
    radius_se: float = Field(..., example=0.405)
    texture_se: float = Field(..., example=1.216)
    perimeter_se: float = Field(..., example=2.833)
    area_se: float = Field(..., example=40.0)
    smoothness_se: float = Field(..., example=0.005)
    compactness_se: float = Field(..., example=0.017)
    concavity_se: float = Field(..., example=0.020)
    concave_points_se: float = Field(..., example=0.010)
    symmetry_se: float = Field(..., example=0.018)
    fractal_dimension_se: float = Field(..., example=0.003)
    worst_radius: float = Field(..., example=17.5)
    worst_texture: float = Field(..., example=28.0)
    worst_perimeter: float = Field(..., example=115.0)
    worst_area: float = Field(..., example=950.0)
    worst_smoothness: float = Field(..., example=0.135)
    worst_compactness: float = Field(..., example=0.260)
    worst_concavity: float = Field(..., example=0.310)
    worst_concave_points: float = Field(..., example=0.114)
    worst_symmetry: float = Field(..., example=0.290)
    worst_fractal_dimension: float = Field(..., example=0.084)

import time

@app.get("/")
def inicio():
    metricas["visitas_raiz"] += 1
    logger.info("Endpoint raiz consultado")
    return {
        "servicio": "Clasificador de Tumores",
        "estado": "activo",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/diagnosticar")
async def diagnosticar(datos: DatosTumor, request: Request):
    inicio_tiempo = time.time()
    metricas["total_predicciones"] += 1
    logger.info(f"Solicitud recibida desde {request.client.host}")

    try:
        entrada = np.array([[
            datos.mean_radius, datos.mean_texture, datos.mean_perimeter,
            datos.mean_area, datos.mean_smoothness, datos.mean_compactness,
            datos.mean_concavity, datos.mean_concave_points, datos.mean_symmetry,
            datos.mean_fractal_dimension, datos.radius_se, datos.texture_se,
            datos.perimeter_se, datos.area_se, datos.smoothness_se,
            datos.compactness_se, datos.concavity_se, datos.concave_points_se,
            datos.symmetry_se, datos.fractal_dimension_se, datos.worst_radius,
            datos.worst_texture, datos.worst_perimeter, datos.worst_area,
            datos.worst_smoothness, datos.worst_compactness, datos.worst_concavity,
            datos.worst_concave_points, datos.worst_symmetry, datos.worst_fractal_dimension
        ]])

        entrada_scaled = scaler.transform(entrada)
        prediccion = modelo.predict(entrada_scaled)[0]
        probabilidades = modelo.predict_proba(entrada_scaled)[0]

        diagnostico = "benigno" if prediccion == 1 else "maligno"
        confianza = round(float(probabilidades.max()) * 100, 2)
        tiempo_ms = round((time.time() - inicio_tiempo) * 1000, 2)

        metricas[f"diagnostico_{diagnostico}"] += 1
        logger.info(f"Diagnostico: {diagnostico} | Confianza: {confianza}% | Tiempo: {tiempo_ms}ms")

        return {
            "diagnostico": diagnostico,
            "confianza_porcentaje": confianza,
            "probabilidades": {
                "maligno": round(float(probabilidades[0]) * 100, 2),
                "benigno": round(float(probabilidades[1]) * 100, 2)
            },
            "tiempo_respuesta_ms": tiempo_ms,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        metricas["errores"] += 1
        logger.error(f"Error en diagnostico: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/metricas")
def ver_metricas():
    logger.info("Consulta de metricas")
    return {"metricas": dict(metricas), "timestamp": datetime.now().isoformat()}

@app.get("/salud")
def salud():
    logger.info("Health check ejecutado")
    return {
        "estado": "saludable",
        "modelo": "GradientBoostingClassifier",
        "precision": "95.61%",
        "total_predicciones": metricas["total_predicciones"]
    }
