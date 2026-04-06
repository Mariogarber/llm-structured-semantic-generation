# Alignment Strategies for Structured Semantic Generation with Large Language Models: Docker-Compose as a Case Study

## 1. Introduction

La generación estructurada mediante Large Language Models (LLMs) representa uno de los retos más relevantes en la investigación actual en modelos generativos. Aunque los LLMs han demostrado una capacidad sobresaliente en generación de texto libre, su comportamiento en tareas que requieren **estructuras formales estrictas y coherencia semántica interna** continúa siendo limitado.

Este Trabajo Fin de Máster propone estudiar la capacidad de distintos enfoques de alineamiento para mejorar la generación estructurada, utilizando como caso de estudio la generación automática de archivos `docker-compose.yaml` a partir de descripciones en lenguaje natural.

El problema no se limita a generar YAML sintácticamente válido, sino que exige:

- Correcta estructura jerárquica.
- Consistencia entre servicios.
- Cumplimiento de dependencias.
- Coherencia semántica (ej. persistencia en bases de datos, exposición de puertos en servicios web).

Se plantea comparar distintos enfoques de alineamiento y entrenamiento para analizar cómo influyen en la generación estructural y semánticamente válida.

---

## 2. Research Questions

1. ¿Hasta qué punto un modelo zero-shot puede generar estructuras formales válidas?
2. ¿Mejora el rendimiento estructural mediante fine-tuning supervisado?
3. ¿Aporta ventajas adicionales el uso de técnicas de alineamiento como RLHF o RLAIF?
4. ¿Existen patrones detectables en el espacio latente asociados a errores estructurales?
5. ¿Puede definirse una métrica formal que evalúe generación estructurada más allá de métricas textuales tradicionales?

---

## 3. Objectives

### 3.1 General Objective

Evaluar el impacto de diferentes estrategias de alineamiento en la capacidad de los LLMs para generar configuraciones estructuradas y semánticamente coherentes.

### 3.2 Specific Objectives

1. Construir un dataset curado de archivos docker-compose.
2. Diseñar métricas estructurales y semánticas formales.
3. Implementar un pipeline experimental comparativo:
   - Zero-shot
   - Fine-tuning supervisado (SFT)
   - RLAIF (mediante reward automático)
4. Analizar el espacio latente del modelo.
5. Evaluar robustez ante ambigüedad y ruido en prompts.

---

## 4. Dataset Construction and Analysis

Se construirá un dataset a partir de repositorios públicos en GitHub que contengan archivos `docker-compose.yaml`.

### 4.1 Proceso

1. Scraping de repositorios públicos.
2. Filtrado automático mediante:
   - Validación YAML.
   - Validación mediante Docker Compose.
3. Eliminación de duplicados.
4. Normalización estructural.

### 4.2 Análisis Exploratorio

- Número medio de servicios por archivo.
- Distribución de:
  - `volumes`
  - `networks`
  - `depends_on`
- Profundidad del árbol YAML.
- Complejidad estructural.

Este análisis permitirá caracterizar formalmente el espacio estructural que el modelo debe aprender.

---

## 5. Methodology

### 5.1 Baseline Model

Se utilizará un LLM open-source (por ejemplo, LLaMA o similar) como modelo base.

Se evaluarán tres configuraciones:

1. Zero-shot
2. Fine-Tuning Supervisado (SFT)
3. Fine-Tuning + RLAIF (Reward Learning from AI Feedback)

---

### 5.2 Reward Model (para RLAIF)

Se diseñará un reward automático basado en:

- Validez sintáctica (parse YAML).
- Validación contra schema.
- Reglas semánticas:
  - Bases de datos deben incluir volúmenes.
  - Servicios web deben exponer puertos.
  - `depends_on` coherente.
  - No conflictos de puertos.

El reward será cuantitativo y podrá integrarse en un esquema de optimización tipo PPO o método simplificado compatible con recursos limitados.

---

## 6. Evaluation Metrics

Dado que BLEU y métricas tradicionales no capturan estructura jerárquica, se definirán métricas específicas.

### 6.1 Structural Metrics

- % YAML válido.
- Tree Edit Distance.
- Exact Match por claves.
- Validación contra esquema formal.

### 6.2 Semantic Metrics

- Cumplimiento de reglas lógicas.
- Penalización por inconsistencias.
- Validación de ejecución en entorno sandbox.

### 6.3 Robustness Metrics

- Sensibilidad ante prompts ambiguos.
- Degradación ante ruido.
- Generalización a configuraciones no vistas.

---

## 7. Latent Space Analysis

Se realizará un análisis del espacio latente del modelo:

- Extracción de embeddings internos.
- Clustering según:
  - Complejidad estructural.
  - Tipo de arquitectura generada.
- Identificación de regiones latentes asociadas a errores estructurales.

Se explorará si los errores presentan patrones estructurales detectables.

---

## 8. Experimental Design

Se plantean los siguientes experimentos:

1. Generación controlada de arquitecturas simples.
2. Generación con dependencias múltiples.
3. Ambigüedad semántica.
4. Robustez ante prompts incompletos.
5. Comparativa cuantitativa entre enfoques.

Cada modelo será evaluado bajo el mismo conjunto de prompts.

---

## 9. Expected Contributions

1. Marco formal para evaluación de generación estructurada.
2. Dataset curado de docker-compose.
3. Comparativa empírica entre estrategias de alineamiento.
4. Propuesta de reward estructural automático.
5. Análisis del comportamiento latente ante restricciones formales.

---

## 10. Limitations

- Recursos computacionales limitados.
- RLHF completo no viable sin infraestructura especializada.
- Generalización fuera del dominio Docker no garantizada.

---

## 11. Future Work

- Extensión a JSON schemas generales.
- Aplicación a Terraform o Kubernetes.
- Incorporación de constraint decoding.
- Integración con validación simbólica.