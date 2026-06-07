# Hipotesis de drift posicional en los estados latentes del cabezal de nivel

Document type: analysis

## Contexto

Esta nota recoge una hipotesis de trabajo surgida durante el analisis de errores
del ultimo experimento `two_head_ordinal_sft` disponible en el repositorio:

```text
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523/
```

El objetivo no es registrar un resultado cerrado, sino dejar escrito el proceso
que llevo a la idea y definir que experimentos permitirian comprobarla. La
hipotesis se situa dentro del marco principal del proyecto: la dificultad de
hacer que un modelo autoregresivo, entrenado para producir una secuencia plana
de tokens, genere una representacion cuya validez depende de una jerarquia
YAML. En este caso, la variable de interes es `level`, es decir, el nivel
jerarquico de cada linea dentro de la representacion normalizada de bloques.

## Observacion inicial durante el analisis de errores

El punto de partida fue una inspeccion cualitativa de pares `pred` / `gold` del
ultimo run. La metrica global ya indicaba un fallo fuerte de la rama
parser-facing: solo `2 / 70` ejemplos parseaban como YAML en la validacion
final. Sin embargo, al mirar ejemplos concretos aparecio un patron mas
especifico que no quedaba completamente explicado por la metrica agregada.

En varios casos donde el contenido textual generado era reconocible, el modelo
fallaba muy pronto en la indentacion. Las primeras lineas del manifiesto
tendian a recibir `level=0` incluso cuando el gold ya habia entrado en
`metadata.name`, `spec.selector`, `matchLabels` o `template.metadata`. Esto se
vio claramente en ejemplos como `q17::question` y `q19::question`: el numero de
bloques coincidia con el gold, y al sustituir solo los niveles predichos por
los niveles gold el YAML pasaba a parsear. Por tanto, el fallo no era
necesariamente que el modelo dejara de emitir bloques validos, sino que la
secuencia de niveles empezaba demasiado plana.

Ese comportamiento llevo a una primera intuicion: tal vez el cabezal no estaba
simplemente confundiendo niveles de forma uniforme, sino que tenia un sesgo
dependiente de la posicion dentro del YAML. Dicho de otra forma, el modelo
parecia comportarse de manera distinta al principio del documento y en las
lineas posteriores.

## Diagnostico por cuartiles del YAML

Para contrastar esa intuicion se calculo una comparacion por cuartiles de
longitud. Cada manifiesto se dividio en cuatro tramos relativos:

```text
Q1 = 0-25%
Q2 = 25-50%
Q3 = 50-75%
Q4 = 75-100%
```

El analisis se hizo principalmente como comparacion de distribuciones, no como
alineamiento linea a linea. Esto es importante porque muchos ejemplos del run
no tienen el mismo numero de bloques predichos y gold. Aun asi, se incluyo un
MAE alineado solo para los `24 / 70` ejemplos donde el numero de lineas
coincidia, con el fin de tener una senal complementaria.

| Cuartil | mean gold | mean pred | delta | gold level 0 | pred level 0 | gold 0-1 | pred 0-1 | gold >=5 | pred >=5 | MAE alineado |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-25% | 0.50 | 0.16 | -0.34 | 62.5% | 89.5% | 90.0% | 95.0% | 0.0% | 0.0% | 0.32 |
| 25-50% | 2.32 | 1.39 | -0.93 | 1.8% | 34.9% | 25.0% | 50.8% | 0.0% | 1.6% | 1.10 |
| 50-75% | 3.42 | 3.04 | -0.37 | 2.6% | 3.5% | 8.7% | 12.1% | 10.4% | 15.4% | 0.76 |
| 75-100% | 3.96 | 3.55 | -0.41 | 1.0% | 1.7% | 8.6% | 5.6% | 30.0% | 18.6% | 0.62 |

El resultado refino la intuicion inicial. En el primer cuartil el gold ya
contiene muchos `level=0`, por lo que cierta abundancia de ceros es esperable.
El problema mas revelador aparece en el segundo cuartil. Entre el 25% y el 50%
del manifiesto, el gold casi ya no esta en `level=0` (`1.8%`), mientras que el
modelo todavia predice `level=0` en un `34.9%` de las lineas. En ese tramo el
MAE alineado sube a `1.10`, y las confusiones principales son transiciones como
`gold 1 -> pred 0`, `gold 2 -> pred 0/1` y `gold 3 -> pred 2`.

Esto sugiere que el fallo no consiste solo en "predecir demasiados ceros al
principio". La lectura mas precisa es que el cabezal parece retrasar la subida
de profundidad. La estructura gold entra pronto en niveles `2`, `3` y `4`, pero
la prediccion permanece durante demasiado tiempo en niveles superficiales. Mas
adelante, en Q3 y Q4, el modelo recupera parte de la profundidad e incluso
predice algunos `level=5`, pero esa recuperacion llega tarde y sigue sin cubrir
bien los niveles mas profundos (`7` y `8` no aparecen).

## Hipotesis planteada

A partir de ese patron surgio la hipotesis principal: el problema podria estar
relacionado con la naturaleza autoregresiva del modelo y con la distribucion de
los estados latentes usados por el MLP de nivel.

En un modelo autoregresivo, cada token se genera condicionado por todo el
contexto anterior. Esto significa que las primeras lineas del manifiesto se
producen con poco contexto autogenerado, mientras que las lineas posteriores se
producen despues de haber acumulado una representacion mas rica del documento
que el propio modelo esta construyendo. Si el estado latente interno se va
enriqueciendo durante la generacion, entonces los vectores que recibe el MLP al
principio del YAML podrian no pertenecer a la misma distribucion que los
vectores que recibe hacia el final.

Esta formulacion inicial debe leerse con cuidado. La idea de que el estado
tardio es "mas rico" no implica automaticamente que el estado temprano sea
intrinsecamente pobre, ni que los niveles esten menos separados al principio
del YAML. La hipotesis mas precisa es de no estacionariedad temporal: el mismo
tipo de decision estructural se toma sobre regiones latentes distintas segun el
momento de la generacion.

La formulacion concreta de la hipotesis es la siguiente:

```text
El MLP de nivel no esta recibiendo una distribucion homogenea de estados
latentes a lo largo de la secuencia. Los estados tempranos y tardios tienen
propiedades diferentes, incluso cuando corresponden al mismo nivel gold. Como
resultado, un cabezal entrenado globalmente puede aprender mejor la separacion
de niveles en unas zonas temporales del documento que en otras. El problema no
seria solo la informacion contenida en el hidden state, sino la forma en que esa
informacion debe leerse en cada momento de la serie.
```

Esta hipotesis no niega el desbalance de niveles del dataset. De hecho, ambos
fenomenos pueden convivir. El primer tramo del YAML contiene naturalmente mas
niveles superficiales, y los niveles profundos tienen menos soporte global. Lo
que se quiere comprobar ahora es algo mas especifico: si, controlando por el
nivel gold, la posicion dentro de la secuencia sigue dejando una huella medible
en el espacio latente. Si esto ocurre, el problema no seria solo una mala
calibracion de thresholds o una frecuencia baja de niveles profundos, sino un
desplazamiento de distribucion entre estados tempranos y tardios.

Tambien hay una segunda posibilidad complementaria. El drift podria existir ya
en modo teacher-forced, cuando se extraen hidden states sobre secuencias gold o
content-only sin dejar que el modelo se equivoque libremente. Pero podria
amplificarse en generacion libre, porque los primeros niveles mal predichos
alteran la estructura del YAML reconstruido y contaminan el contexto que el
modelo usa para predecir las lineas siguientes. En ese caso habria dos efectos:
un sesgo posicional propio de la representacion latente y una propagacion de
errores durante la generacion.

## Lectura como serie temporal autoregresiva

Una forma mas clara de formular el problema es tratar la generacion del YAML
como una serie temporal. En cada paso, el modelo no solo produce texto, sino que
tambien genera el estado latente desde el que el cabezal estructural debe
predecir el nivel:

```text
token_t -> hidden_t -> level_t
```

En el caso de la representacion por lineas, varios tokens contribuyen a una
misma linea, pero la intuicion temporal se mantiene. La prediccion de `level`
no es una clasificacion aislada de una linea abstracta, sino una decision
tomada en un punto concreto de una secuencia autoregresiva. Por tanto,
`hidden_t` debe interpretarse junto con el momento de generacion en el que
aparece.

Desde esta perspectiva, el YAML no es solo una lista de pares
`(line_text, level)`. Es una trayectoria:

```text
(hidden_1, level_1), (hidden_2, level_2), ..., (hidden_T, level_T)
```

A medida que avanza la trayectoria, cambia el contexto acumulado por el modelo:
ya se han generado campos, claves, recursos, contenedores o bloques anidados.
La distribucion de los estados latentes puede desplazarse de forma gradual o
por tramos. Los experimentos por cuartiles detectan que existe ese cambio, pero
no identifican todavia el punto exacto ni la forma continua del desplazamiento.
Por eso los cuartiles deben entenderse como una primera discretizacion de la
serie, no como fronteras estructurales definitivas.

Esta lectura tambien evita una solucion demasiado directa pero problematica:
crear clases combinadas del tipo `level x position_bin`. Esa estrategia
codificaria explicitamente el momento de generacion, pero multiplicaria el
numero de clases y agravaria el desbalance que ya existe en los niveles
profundos. En vez de aumentar el espacio de etiquetas, la direccion mas
razonable es condicionar el cabezal de `level` con informacion temporal o
posicional.

La hipotesis revisada queda entonces asi:

```text
La posicion temporal de la linea actua como una condicion de dominio para leer
el hidden state. Un mismo `level` puede estar representado por regiones latentes
distintas al principio y al final del YAML. Por tanto, el cabezal de nivel no
deberia tratar todos los hidden states como muestras intercambiables de una
unica distribucion estacionaria.
```

Esta formulacion es mas prudente que afirmar que la representacion inicial es
mas pobre. Los resultados disponibles muestran diferencia de distribucion y no
estacionariedad temporal. Para afirmar pobreza representacional harian falta
pruebas adicionales, por ejemplo menor separabilidad entre niveles, menor
informacion mutua con `level` o peor rendimiento de probes controlando por
soporte y desbalance.

## Experimentos propuestos

La hipotesis debe comprobarse con experimentos que separen posicion, nivel gold
y calidad de la generacion. Los experimentos siguientes se plantean como una
secuencia razonable.

### 1. Clasificador de cuartil desde hidden states

El primer experimento consiste en usar los hidden states ya extraidos por el
pipeline `latent_level_probe`, pero cambiando la etiqueta supervisada. En vez
de entrenar un probe para predecir `level`, se entrenaria un clasificador para
predecir el cuartil posicional de la linea:

```text
hidden_state -> {Q1, Q2, Q3, Q4}
```

Si el clasificador predice el cuartil claramente por encima del azar (`25%`),
habria evidencia de que el espacio latente contiene informacion posicional. Sin
embargo, esta prueba por si sola no basta, porque el cuartil esta correlacionado
con el nivel: Q1 contiene mas `level=0/1` y Q4 contiene mas niveles profundos.

Por eso, el control importante es repetir el experimento dentro de cada nivel
con soporte suficiente:

```text
hidden_state | gold level = 2 -> cuartil
hidden_state | gold level = 3 -> cuartil
hidden_state | gold level = 4 -> cuartil
hidden_state | gold level = 5 -> cuartil
```

Si dentro del mismo nivel gold sigue siendo posible predecir el cuartil, la
evidencia a favor de drift posicional seria mucho mas fuerte. En ese caso,
`level=3` temprano y `level=3` tardio no serian intercambiables para el cabezal.

### 2. Transferencia cruzada entre cuartiles

El segundo experimento prueba directamente si un probe entrenado en una zona de
la secuencia generaliza a otra. Se entrenarian probes de nivel separados por
cuartil:

```text
train Q1 -> eval Q1, Q2, Q3, Q4
train Q2 -> eval Q1, Q2, Q3, Q4
train Q3 -> eval Q1, Q2, Q3, Q4
train Q4 -> eval Q1, Q2, Q3, Q4
```

La metrica principal deberia ser MAE de `level`, acompanada de balanced
accuracy y matrices de confusion, porque el soporte por nivel cambia mucho
entre cuartiles. Una matriz de transferencia con buena diagonal y mala
generalizacion fuera del cuartil indicaria que los estados latentes no son
estacionarios a lo largo de la secuencia.

La forma esperada del resultado, si la hipotesis es correcta, seria:

```text
          eval Q1  eval Q2  eval Q3  eval Q4
train Q1   mejor    medio    peor     peor
train Q4   peor     medio    medio    mejor
```

Este experimento deberia hacerse con cuidado para no confundir drift latente
con ausencia de clases. Si Q1 no contiene niveles profundos, un probe entrenado
solo en Q1 no puede aprenderlos. Por tanto, seria necesario registrar el
soporte por nivel de cada split y, cuando sea posible, hacer una version
balanceada por `level`.

### 3. Distancias de distribucion latente por cuartil y nivel

El tercer experimento no entrena necesariamente un clasificador, sino que mide
la geometria de los estados. Para cada nivel con soporte suficiente se
calcularian centroides por cuartil:

```text
centroid(level=3, Q1)
centroid(level=3, Q2)
centroid(level=3, Q3)
centroid(level=3, Q4)
```

Despues se compararian esos centroides mediante distancia coseno o distancia
euclidea sobre vectores estandarizados. Tambien se podria usar una medida de
distribucion como MMD si se quiere evitar depender solo de la media. La senal
que se buscaria es que, para un mismo `level`, la distancia entre Q1 y Q4 sea
mayor que la variabilidad interna esperable.

Este analisis puede acompanarse de PCA, t-SNE o UMAP, coloreando por cuartil y
facetando por `level`. La visualizacion no deberia ser la prueba principal,
pero si puede ayudar a interpretar si el desplazamiento es suave, si hay
clusters separados o si el efecto aparece solo en algunos niveles.

### 4. Comparacion entre teacher forcing y generacion libre

El cuarto experimento separa dos escenarios que conviene no mezclar. En modo
teacher-forced, los hidden states se extraen sobre una secuencia controlada,
normalmente derivada del gold o del formato content-only sin columna `level`.
En generacion libre, los estados se producen durante la salida real del modelo,
con sus propios errores acumulados.

La comparacion seria:

```text
teacher-forced content_blocks -> drift posicional
free-generation validation -> drift posicional
```

Si el drift aparece en ambos, hay evidencia de que la distribucion latente
cambia de forma natural a lo largo de la secuencia autoregresiva. Si aparece
mucho mas fuerte en generacion libre, entonces la explicacion deberia incluir
propagacion de errores: las primeras decisiones incorrectas del modelo cambian
el contexto y hacen que las siguientes predicciones de nivel se apoyen en una
historia ya deformada.

### 5. Analisis de calibracion del `ordinal_score` por cuartil

En el run ordinal actual cada bloque predicho incluye un `ordinal_score`. Esto
permite una prueba adicional, especifica del cabezal ordinal. Se puede analizar
la distribucion de `ordinal_score` por cuartil y compararla con los thresholds
finales aprendidos. La pregunta seria si en Q1 y Q2 los scores quedan
sistematicamente por debajo de los cortes necesarios para producir niveles
mayores, incluso cuando el gold ya exige `level=2`, `level=3` o `level=4`.

Este analisis conectaria directamente la hipotesis latente con el fallo
observado en la salida:

```text
hidden state temprano -> ordinal_score bajo -> level predicho superficial
```

Si Q2 concentra scores demasiado bajos para lineas cuyo gold esta en `2/3/4`,
entonces el problema no estaria solo en los thresholds globales, sino en la
forma en que el MLP proyecta estados tempranos frente a estados tardios.

## Criterios de interpretacion

La hipotesis quedaria reforzada si se observan tres senales a la vez. Primero,
que el cuartil pueda predecirse desde el hidden state incluso dentro de un mismo
level gold. Segundo, que los probes entrenados en un cuartil pierdan rendimiento
al evaluarse en otros cuartiles, especialmente en direccion Q4 -> Q1 o Q1 ->
Q4. Tercero, que los centroides o distribuciones latentes del mismo nivel se
desplacen de forma consistente a lo largo de la secuencia.

En cambio, si al controlar por nivel desaparece casi toda la senal de cuartil,
la explicacion deberia volver hacia el desbalance estructural del dataset y la
frecuencia real de cada nivel por posicion. Si los probes cruzados generalizan
bien entre cuartiles, entonces el problema del ultimo run seria mas compatible
con mala calibracion del cabezal ordinal, thresholds inadecuados o error de
contenido, pero no con un drift latente fuerte.

La utilidad de esta hipotesis es que no cambia el objetivo principal del
proyecto, pero si puede orientar mejor las siguientes decisiones de modelado. Si
el drift posicional existe, podria tener sentido introducir senales explicitas
de posicion relativa, normalizacion o calibracion por tramo, o estrategias de
entrenamiento que obliguen al cabezal a generalizar entre estados tempranos y
tardios. Si no existe, entonces conviene centrar el esfuerzo en otros factores
ya identificados, como el desbalance de niveles profundos, la eleccion del loss
ordinal o la estabilidad de la generacion de contenido.

## Implicaciones de modelado

La consecuencia practica de esta lectura temporal no es necesariamente crear
mas clases. De hecho, separar la etiqueta en combinaciones `level x bin`
probablemente agravaria el problema de imbalance, porque cada nivel profundo
quedaria repartido entre menos ejemplos por tramo.

Las alternativas mas coherentes con la hipotesis son mecanismos que mantengan
el mismo espacio de salida, pero den al cabezal informacion sobre el momento en
que se esta tomando la decision:

- anadir `relative_position` como feature continua junto al hidden state;
- codificar la posicion con embeddings o Fourier features antes del MLP;
- usar gating o FiLM para que la posicion module la lectura del hidden state;
- anadir un sesgo posicional sobre los logits o sobre el `ordinal_score`;
- calibrar thresholds por posicion solo como ablation controlada, no como
  solucion principal cerrada.

Estas ideas deben tratarse como propuestas de experimento, no como decisiones
cerradas del modelo. Su funcion es comprobar si el cabezal mejora cuando deja
de asumir que todos los estados latentes pertenecen a una unica distribucion
temporalmente estacionaria.

### Primera variante implementable: concat final causal

La primera prueba operacional debe ser deliberadamente simple. En vez de usar
FiLM o gating desde el principio, se propone un cabezal posicional con
concatenacion final:

```text
hidden -> LayerNorm -> Linear(hidden, 512) -> GELU -> Dropout(0.10)
       -> Linear(512, 64) -> GELU -> LayerNorm(64)

line_position -> sin/cos absolute causal encoding, 16 dims -> LayerNorm(16)

concat(hidden_64, position_16) -> Linear(80, 1) -> ordinal_score
```

La posicion usada es causal: `line_position` es el indice de linea conocido en
ese momento, no una fraccion sobre la longitud final del YAML. Las frecuencias
iniciales son:

```text
1, 2, 4, 8, 16, 32, 64, 128
```

El positional encoding no usa dropout. El objetivo de esta variante no es
demostrar que sea la solucion final, sino comprobar si una senal temporal
sencilla mejora la lectura del hidden state sin multiplicar las clases ni
empeorar directamente el imbalance.

### Auditoria posterior a Positional Head V1

La primera lectura intermedia del run posicional sugiere que el modelo no esta
ignorando la posicion. En `checkpoint-step-80`, el `ordinal_score` aumenta de
forma clara con el indice de linea generado: los scores medios pasan de valores
muy bajos en Q1 a valores menos negativos en Q4. Esto encaja con la idea de que
la rama posicional esta activa.

Sin embargo, el patron observado no demuestra todavia que la posicion este
ayudando en el sentido deseado. La arquitectura `final_concat` permite que la
posicion entre justo antes de la ultima proyeccion lineal:

```text
ordinal_score = W_hidden * hidden_64 + W_pos * position_16 + b
```

En esa forma, la posicion puede funcionar como un sesgo temporal aditivo. Es
decir, el cabezal puede aprender una regla gruesa del tipo "lineas tempranas
producen scores bajos; lineas tardias producen scores mas altos" sin aprender
necesariamente a reinterpretar el hidden state segun el momento de generacion.
Esto explicaria por que Q1 sigue comprimido a `level=0` y Q2 mantiene demasiada
masa en `level=0`, aunque algunas lineas tempranas ya requieran niveles
profundos.

La siguiente auditoria debe separar tres posibilidades:

- la posicion domina el cabezal como shortcut temporal;
- la posicion apenas afecta y el fallo sigue viniendo del hidden/proyector;
- la posicion ayuda, pero el punto de inyeccion es demasiado tardio y lineal.

Para ello se propone comparar el mismo checkpoint con positional encoding real,
positional encoding anulado y positional encoding barajado, siempre sobre los
mismos `content_blocks` generados. Tambien se propone un baseline `position
only` para medir cuanta senal de `level` puede explicarse solo por la posicion
absoluta o por bins causales de posicion.

Si esta lectura se confirma, la siguiente variante no deberia aumentar
simplemente la dimension del embedding posicional. La direccion mas adecuada es
un mecanismo de condicionamiento feature-wise, como FiLM o gating, en el que la
posicion module las features internas del MLP antes de la proyeccion ordinal
final.

## Artefactos relacionados

```text
scripts/audit_positional_head_v1.py
scripts/train_kubernetes_two_head_ordinal_positional_sft.py
scripts/train_kubernetes_two_head_ordinal_film_positional_sft.py
docs/analysis/latent/POSITIONAL_CONDITIONING_NEXT_STEPS_2026-05-27.md
docs/analysis/latent/LATENT_LEVEL_PROBE_V1.md
docs/analysis/latent/runs/LATENT_LEVEL_PROBE_RESULTS_2026-05-13.md
docs/experiments/two_head_sft/runs/TWO_HEAD_ORDINAL_DENSITY_V2_CENTERED_GAP05_MLP3_THRESHOLD_LR50_RUN_20260523_ANALYSIS.md
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523/validation_predictions.jsonl
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523/metrics.json
```
