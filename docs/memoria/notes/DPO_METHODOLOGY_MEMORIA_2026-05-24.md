# Notas para memoria: metodologia DPO sobre `serialized_sft`

Document type: memoria note

## Punto de partida

La fase de DPO se plantea despues de observar dos resultados complementarios. Por
un lado, el modelo `serialized_sft` funciona como una referencia supervisada
fuerte: aprende a emitir la representacion `blocks_tsv_v1`, mantiene una tasa
muy alta de reconstruccion YAML parseable y mejora de forma clara la adecuacion
al prompt respecto al baseline. Por otro lado, las ramas con cabezal explicito
para `level` siguen siendo interesantes desde el punto de vista investigador,
pero su comportamiento actual es demasiado inestable como sistema generativo
final. La caida de parseabilidad hace que no sean, por ahora, la mejor base para
una fase de alineamiento.

Esta situacion cambia ligeramente la pregunta experimental. Ya no se trata de
demostrar que el modelo puede aprender la superficie estructurada basica, porque
el SFT serializado ya lo ha mostrado en validacion. La pregunta pasa a ser mas
fina: si un modelo que ya genera salidas estructuralmente razonables puede ser
empujado hacia manifiestos mas adecuados al prompt y mas coherentes con buenas
practicas de Kubernetes mediante preferencias automaticas. En este sentido, DPO
no sustituye al SFT, sino que se coloca sobre el como una fase de refinamiento.

## Por que DPO es la primera tecnica elegida

DPO es especialmente adecuado para este punto del proyecto porque trabaja con
pares de salidas preferidas y rechazadas sin exigir un modelo de recompensa
separado ni una fase de aprendizaje por refuerzo online. Esta propiedad encaja
con las restricciones practicas del repositorio: existe un SFT fuerte que puede
actuar como politica de referencia, existe un evaluador automatico que produce
senales de validez estructural y de dominio, y no hay todavia una recompensa
formal suficientemente validada como para justificar PPO.

Formalmente, DPO parte de una idea comun con RLHF: una politica no solo debe
maximizar una recompensa, sino hacerlo sin alejarse demasiado de una politica de
referencia. Esa restriccion es importante porque la politica de referencia, en
este caso `serialized_sft`, ya contiene comportamientos que se quieren
conservar: el formato `blocks_tsv_v1`, la interaccion con el parser, la
estructura de lineas y niveles, y una buena parte de la adecuacion al prompt. Si
el alineamiento destruyera esa base, el resultado no seria una mejora real del
sistema.

La formulacion habitual de RLHF regularizado puede escribirse como:

```text
max_pi E[x,y~pi] [r(x,y)] - beta * D_KL(pi(.|x) || pi_ref(.|x))
```

La contribucion de DPO consiste en evitar el entrenamiento explicito del modelo
de recompensa. Bajo un modelo de preferencias de Bradley-Terry, la probabilidad
de que una salida `y_w` sea preferida a otra `y_l` depende de la diferencia de
recompensas:

```text
P(y_w > y_l | x) = sigma(r*(x,y_w) - r*(x,y_l))
```

Al expresar la recompensa implicita en funcion del cociente entre la politica
entrenable y la politica de referencia, la diferencia anterior puede optimizarse
directamente con:

```text
L_DPO =
  - log sigma(
      beta * [
        log pi_theta(y_w|x) - log pi_ref(y_w|x)
      - log pi_theta(y_l|x) + log pi_ref(y_l|x)
      ]
    )
```

Esta perdida aumenta la probabilidad relativa de la salida preferida frente a la
rechazada, pero lo hace comparando siempre con la politica de referencia. En el
contexto de este trabajo, esta caracteristica es importante porque el objetivo
no es que el modelo explore cualquier forma de maximizar una metrica proxy, sino
que mejore sobre el comportamiento ya estable del SFT.

## Por que PPO queda como extension

PPO es una tecnica mas general y, en ciertas condiciones, puede superar a DPO.
La bibliografia comparativa entre DPO y PPO muestra que PPO no debe descartarse
por inferioridad teorica. Sin embargo, tambien muestra que su rendimiento depende
de una implementacion cuidadosa: control de KL, normalizacion de ventajas,
tamanos de batch suficientemente grandes, estabilidad del modelo de recompensa y
rollouts online. Para un proyecto academico con recursos limitados, esa carga
experimental introduce muchos grados de libertad nuevos.

En este trabajo, el problema principal no es todavia entrenar una politica
online contra una recompensa madura. El problema es construir una primera senal
de preferencia automatica que sea trazable y que permita comprobar si hay margen
de mejora despues del SFT. Por eso PPO se deja como extension futura: tendria
sentido si DPO no fuera suficiente y si las senales automaticas de recompensa
demostraran ser estables. Antes de eso, PPO mezclaria la pregunta sobre el
alineamiento con otra pregunta adicional sobre ingenieria de recompensa.

## Construccion de preferencias automaticas

El dataset de DPO se construira generando varias salidas candidatas para cada
prompt del split de entrenamiento. Todas las candidatas procederan del propio
`serialized_sft`, con diferentes semillas o configuraciones moderadas de
decodificacion. Esta decision reduce el desplazamiento de distribucion entre la
politica que produjo los datos de preferencia y la politica de referencia que
aparece en la perdida DPO.

Cada candidata se evaluara con el mismo stack que ya se usa en baseline y SFT.
Primero se comprueba si la salida puede interpretarse como `blocks_tsv_v1`.
Despues se reconstruye el YAML mediante el parser determinista. Finalmente se
calculan las metricas estructurales, de adecuacion al prompt y de validez de
dominio Kubernetes. Solo las candidatas que superan la frontera estructural
pueden ser elegidas como salidas preferidas.

La funcion de ranking combina dos intuiciones. La primera es que el manifiesto
debe responder al prompt. Por eso `prompt_requirement_f1` tiene el mayor peso.
La segunda es que, entre salidas igualmente fieles al prompt, se debe preferir
la que se comporta mejor como manifiesto Kubernetes. Para ello se usan
`kubernetes_domain_validity_score`, `kubernetes_domain_gate_pass` y completitud
de campos requeridos. La metrica `kubernetes_domain_gate_pass` es especialmente
valiosa porque resume el paso por el gate actual de buenas practicas, pero no se
usa como unico objetivo porque es demasiado binaria. Si casi todas las salidas
fallan el gate completo, no proporcionaria una senal suficientemente gradual.

El score propuesto es:

```text
score =
  1.00 * prompt_requirement_f1
+ 0.75 * kubernetes_domain_validity_score
+ 0.50 * required_field_complete_resource_rate
+ 0.25 * level_exact_match_rate
+ 0.25 * kubernetes_domain_gate_pass
- penalties
```

Para reducir ruido, solo se conservaran pares donde la diferencia entre la
salida preferida y la rechazada supere un margen minimo:

```text
score(chosen) - score(rejected) >= 0.15
```

Este margen es importante porque DPO puede degradarse si se entrena con
preferencias ambiguas o mal etiquetadas. En una tarea como esta, donde las
metricas son automaticas y aproximadas, no conviene forzar pares cuando dos
salidas son practicamente indistinguibles.

## Papel de las metricas guia

Las dos metricas mas importantes para la fase de DPO son
`prompt_requirement_f1` y las senales KDV, especialmente
`kubernetes_domain_validity_score` y `kubernetes_domain_gate_pass`. La primera
protege la fidelidad a la peticion del usuario; las segundas capturan hasta que
punto el YAML generado se comporta como una configuracion Kubernetes razonable
segun los checks automaticos actuales.

El equilibrio entre ambas es central. Una salida puede ser muy limpia desde el
punto de vista de buenas practicas y, aun asi, no responder al prompt. Tambien
puede cubrir el prompt de forma superficial pero producir un manifiesto pobre o
incompleto. La metodologia no debe escoger entre estas dos dimensiones como si
fueran excluyentes. Precisamente por eso el score combina ambas y la evaluacion
posterior las reporta por separado.

## Riesgo de alignment tax

La fase de DPO debe evaluarse teniendo en cuenta el riesgo de alignment tax. Al
optimizar una preferencia, el modelo puede perder parte de las capacidades que
habia adquirido durante SFT. En este caso, la perdida no se mediria como una
caida generica de rendimiento linguistico, sino como deterioro de propiedades
que son esenciales para el sistema: parseabilidad YAML, cumplimiento del
contrato de bloques, calidad del texto de linea y coherencia de los niveles.

Por eso el resultado DPO no se aceptara automaticamente por mejorar una metrica
de dominio. Si aumenta `kubernetes_domain_gate_pass_rate` pero baja de forma
clara `yaml_parse_success_rate` o `prompt_requirement_f1`, el resultado se
interpretara como sobreoptimizacion de una senal proxy. La pregunta no es si DPO
puede mover el modelo, sino si puede moverlo en una direccion util sin romper lo
que ya funcionaba.

## Interpolacion y frente de Pareto

Para estudiar ese equilibrio se evaluara una interpolacion entre el modelo SFT y
el modelo DPO:

```text
theta_alpha = (1 - alpha) theta_sft + alpha theta_dpo
```

En la practica, como el ajuste se realiza mediante LoRA, la interpolacion se
aplicara preferentemente sobre adapters compatibles:

```text
adapter_alpha = (1 - alpha) adapter_sft + alpha adapter_dpo
```

El barrido de `alpha` permitira observar si existe un frente de Pareto entre
alineamiento y retencion. En un extremo esta el SFT original, que conserva la
maxima estabilidad conocida. En el otro esta el checkpoint DPO, que representa
la maxima aplicacion de la senal de preferencia. Entre ambos puede aparecer un
punto mas interesante que cualquiera de los dos extremos: un modelo que adopta
parte de la mejora de DPO sin pagar todo el coste de degradacion estructural.

Este analisis es metodologicamente importante porque evita una conclusion
demasiado binaria. DPO no tiene que ser simplemente "mejor" o "peor" que SFT. La
pregunta puede formularse como un compromiso medible: cuanta alineacion
Kubernetes/prompt se gana por cada unidad de estabilidad estructural que se
pierde.

## Evaluacion

La evaluacion principal comparara:

```text
baseline
serialized_sft
serialized_sft_dpo
```

En validacion se usaran varias configuraciones de `beta`:

```text
beta in {0.05, 0.10, 0.20}
```

Despues se seleccionara el mejor checkpoint y, si procede, el mejor valor de
interpolacion `alpha`. Solo entonces se evaluara en test. Esta separacion es
importante para no convertir el test en una herramienta de seleccion.

Las metricas se dividiran en dos grupos. El primer grupo mide alineamiento con
la preferencia:

```text
average_prompt_requirement_f1
prompt_requirement_exact_match_rate
average_kubernetes_domain_validity_score
kubernetes_domain_gate_pass_rate
average_required_field_complete_resource_rate
```

El segundo grupo mide retencion estructural:

```text
structured_output_parse_success_rate
yaml_parse_success_rate
block_parse_success_rate
average_line_text_f1
average_level_exact_match_rate
average_level_mae
line_count_match_rate
```

El criterio minimo de exito sera mejorar `average_prompt_requirement_f1` o
`average_kubernetes_domain_validity_score` sin reducir `yaml_parse_success_rate`
mas de `0.02` absoluto respecto a `serialized_sft`. Este umbral conserva la
prioridad del proyecto: no basta con obtener manifiestos que parezcan mejores
segun una metrica si se pierde la capacidad de reconstruir YAML valido.

## Lectura esperada de resultados

Si DPO mejora la adecuacion al prompt y la validez Kubernetes sin degradar la
parseabilidad, se podra defender que las preferencias automaticas son una fase
post-SFT util para generacion estructurada. Si la mejora aparece solo despues de
interpolar con SFT, el resultado seguira siendo positivo, pero se interpretara
como evidencia de un tradeoff de alineamiento. Si DPO no mejora, el resultado
tambien sera informativo: indicara que el SFT serializado ya absorbe buena parte
de la senal disponible o que las metricas automaticas actuales no son
suficientemente ricas para guiar una fase de preferencias.

En cualquier caso, esta fase debe presentarse con prudencia. No demuestra una
correccion semantica completa de Kubernetes ni sustituye una validacion oficial
contra schema. Lo que evalua es mas concreto y mas defendible: si un modelo ya
estable en la superficie estructurada puede ser refinado mediante preferencias
automaticas hacia salidas mas adecuadas y mas validas segun el conjunto de
checks disponibles en el proyecto.

