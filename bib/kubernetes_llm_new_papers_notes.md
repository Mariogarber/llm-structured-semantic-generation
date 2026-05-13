# Nuevos papers sobre LLMs, YAML y Kubernetes

Notas de lectura para integrar en la memoria. Los PDFs asociados se han descargado en esta misma carpeta con prefijos `35` a `38`.

## 35. CloudEval-YAML

**Archivo:** `35_cloudeval_yaml_cloud_native_configuration_generation.pdf`  
**Cita sugerida:** `\citep{xu2024cloudevalyaml}`  
**Uso recomendado:** estado del arte, secciones de generación YAML cloud-native y evaluación.

CloudEval-YAML es especialmente útil para justificar que la generación de YAML cloud-native no debe evaluarse solo con similitud textual. El benchmark combina descripciones en lenguaje natural, posible contexto YAML, YAML de referencia y scripts de test funcionales. Esta estructura refuerza la idea de que un manifiesto puede parecer cercano al objetivo y aun así fallar cuando se ejecuta o valida funcionalmente.

Conclusiones útiles:

- La evaluación incluye métricas textuales, métricas conscientes de YAML y unit tests funcionales. Esto encaja directamente con la separación entre parseabilidad, estructura y validez de dominio que se propone en el TFM.
- El paper señala que BLEU y otras métricas textuales pueden penalizar diferencias irrelevantes para YAML, como el orden de objetos, y por tanto no capturan bien la corrección real.
- Incluso modelos fuertes cometen errores simples en configuraciones cloud-native; esto ayuda a motivar mecanismos de control estructural y validación automática.
- Few-shot prompting no produce mejoras claras en el benchmark, mientras que multi-sample generation puede mejorar resultados si existe un mecanismo de selección o test.
- La evaluación funcional es cara, de modo que el paper también motiva el uso de señales proxy o validadores parciales cuando no se puede ejecutar todo el entorno.

## 36. KGen: from Intent to Kubernetes Manifest

**Archivo:** `36_kgen_intent_to_kubernetes_manifest.pdf`  
**Cita sugerida:** `\citep{angi2025kgen}`  
**Uso recomendado:** estado del arte, sección sobre generación de manifiestos Kubernetes desde intención natural.

KGen conecta de forma muy directa con el TFM porque estudia la generación de manifiestos Kubernetes a partir de intenciones en lenguaje natural. Su enfoque no es exactamente el mismo, pero sí comparte la preocupación central: traducir una intención humana a una configuración jerárquica válida.

Conclusiones útiles:

- La generación Kubernetes desde lenguaje natural es sensible al modelo y al número de ejemplos few-shot; más ejemplos no siempre mejoran el resultado.
- En modelos especializados como Mixtral-8x7B o Prometheus-8x7B-v2.0, aumentar ejemplos puede mejorar la calidad, pero en modelos generalistas como Llama3-8B o Llama3-70B puede reducir el número de manifiestos válidos.
- El paper valida primero sintaxis YAML y después compara los manifiestos reconstruidos con los originales, lo que apoya una evaluación por capas.
- El pipeline usa plantillas y placeholders tipo Helm para preservar la estructura jerárquica de los objetos Kubernetes. Esto es una conexión fuerte con la hipótesis del TFM: la jerarquía no es un detalle superficial, sino parte del objeto que se genera.
- La conclusión más citable para el TFM es que la generación estructurada específica de dominio requiere análisis empírico del setup; no basta con asumir que un modelo más grande o más contexto resuelve el problema.

## 37. Migrating Existing Container Workload to Kubernetes

**Archivo:** `37_migrating_container_workload_to_kubernetes.pdf`  
**Cita sugerida:** `\citep{ueno2024migrating}`  
**Uso recomendado:** estado del arte, sección Kubernetes como dominio y evaluación.

Este paper estudia la conversión de especificaciones Docker Compose a manifiestos Kubernetes mediante LLMs. Es útil para diferenciar tu trabajo de una migración formal desde Compose, pero también para reforzar que Kubernetes exige evaluación específica y no solo generación textual.

Conclusiones útiles:

- Kubernetes supone una barrera para desarrolladores por su complejidad y por la distancia entre especificaciones más simples, como Docker Compose, y manifiestos Kubernetes completos.
- Los autores proponen evaluar los manifiestos generados por LLMs en términos de corrección, groundedness respecto a la entrada, consistencia y mantenibilidad.
- Las herramientas estáticas o de linting no bastan para saber si una salida satisface la especificación de entrada.
- Los LLMs pueden cubrir huecos simples de especificación y producir resultados razonables, pero fallan más en entradas atípicas o con intención poco clara.
- Los comentarios y elementos de legibilidad tienden a perderse, lo que permite recordar que una salida técnicamente válida no siempre es igual a una salida mantenible.

## 38. Smells in ChatGPT-generated Kubernetes manifests

**Archivo:** `38_chatgpt_kubernetes_manifest_smells.pdf`  
**Cita sugerida:** `\citep{zhang2024generativeSmells}`  
**Uso recomendado:** estado del arte, sección Kubernetes como dominio de evaluación estructurada.

Este paper es una cita muy útil para defender que los manifiestos generados por LLMs deben pasar por control de calidad específico del dominio. No basta con que el YAML sea parseable ni con que el texto parezca razonable.

Conclusiones útiles:

- El estudio encuentra smells en manifiestos Kubernetes generados por ChatGPT, incluyendo problemas de seguridad y red.
- En el resumen y conclusión, los autores reportan que el 35,8% de los 98 manifiestos analizados contiene al menos una instancia de smell.
- Los smells más frecuentes son la ausencia de requisitos de CPU y memoria, ambos relevantes para seguridad y robustez operacional.
- Los objetos afectados principales son `Deployment` y `Service`.
- Los autores recomiendan aplicar actividades de quality assurance, como análisis estático, antes de usar manifiestos generados por ChatGPT.
- Para el TFM, esta cita justifica separar validez sintáctica, estructura, validez Kubernetes y semántica aproximada: un manifiesto puede pasar checks superficiales y seguir incluyendo defectos operativos.

