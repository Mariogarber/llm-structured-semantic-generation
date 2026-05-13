# Latent Level Probe Results 2026-05-13

## Indice rapido

- [Contexto](#contexto)
- [Configuracion del experimento](#configuracion-del-experimento)
- [Arquitectura del MLP usado](#arquitectura-del-mlp-usado)
- [Resultados principales](#resultados-principales)
- [Comparacion entre `record_prefix_state` y `line_prefix_state`](#comparacion-entre-record_prefix_state-y-line_prefix_state)
- [Analisis de matrices de confusion](#analisis-de-matrices-de-confusion)
- [Discusion sobre `line_index`](#discusion-sobre-line_index)
- [Fallos especificos de `record_prefix_state`](#fallos-especificos-de-record_prefix_state)
- [Como mejorar el probe MLP](#como-mejorar-el-probe-mlp)
- [Conclusiones](#conclusiones)

## Contexto

Este informe documenta el primer run completo del experimento
`latent_level_probe` sobre Kubernetes v1:

```text
latent-level-probe-real-full-20260513-1528
```

El objetivo del experimento no era entrenar todavia el modelo `two_head_sft`,
sino comprobar una pregunta previa: si los hidden states del modelo base ya
contienen informacion recuperable sobre el `level` de cada linea YAML. Esta
pregunta es importante porque el problema central del proyecto no consiste solo
en generar texto parecido a YAML. Un manifiesto Kubernetes se serializa como
texto, pero su validez depende de una jerarquia de campos, listas, recursos y
relaciones. En la representacion estructural del repositorio, esa jerarquia se
resume linea a linea mediante la variable `level`.

Por tanto, este probe funciona como un diagnostico antes de introducir una
cabeza explicita de nivel. Si un clasificador ligero puede recuperar `level`
desde un hidden state fijo, hay evidencia de que el modelo ya organiza parte de
la informacion jerarquica en su espacio interno. Si solo puede hacerlo un MLP
no lineal, la informacion podria estar presente pero no de forma linealmente
separable. Y si los probes fallaran, la lectura seria distinta: `two_head_sft`
no solo tendria que leer una senal existente, sino contribuir a formar una
representacion mas estructural durante el SFT.

Es importante separar esta variable de otras metricas del proyecto. En este
documento, `level` significa el nivel jerarquico o de indentacion de una linea
YAML en la representacion normalizada. No se refiere al "nivel 5" de la metrica
de evaluacion Kubernetes usada para valorar manifiestos completos.

## Configuracion del experimento

El run uso el modelo base local:

```text
model/qwen2.5-7b-instruct-4bit/
```

No se cargo ningun adapter LoRA. La capa analizada fue la ultima capa decoder
del modelo (`hidden_layer=-1`). El experimento se ejecuto con `train` y
`validation`; el split `test` no se uso, de acuerdo con la decision de
reservarlo para la evaluacion final de modelos candidatos.

| split | unidades | lineas |
|---|---:|---:|
| train | 426 | 8336 |
| validation | 70 | 1800 |

La distribucion de niveles en validation fue:

| level | lineas |
|---:|---:|
| 0 | 324 |
| 1 | 294 |
| 2 | 230 |
| 3 | 328 |
| 4 | 450 |
| 5 | 114 |
| 6 | 44 |
| 7 | 4 |
| 8 | 12 |

El input usado para extraer hidden states fue una version sin leakage del target
`blocks_tsv_v1`. El modelo ve `document_index`, `line_index` y `line_text`, pero
no ve la columna gold `level`. El `level` se conserva solo como etiqueta
supervisada para entrenar y evaluar los probes.

Se compararon cinco estrategias de lectura:

| estrategia | que ve |
|---|---|
| `record_prefix_state` | prompt y lineas previas; no ve informacion de la linea actual |
| `line_prefix_state` | `document_index`, `line_index` y el marcador previo a `line_text`; no ve el contenido de la linea |
| `line_first_token` | primer token del contenido de la linea |
| `line_last_token` | ultimo token del contenido de la linea |
| `line_mean` | media de los hidden states de los tokens de la linea |

Y cuatro tipos de probe:

| probe | funcion |
|---|---|
| `majority` | baseline que predice siempre el nivel mas frecuente |
| `previous_level` | baseline estructural que predice el nivel anterior dentro del mismo sample |
| `linear` | regresion logistica multinomial sobre el vector latente |
| `mlp` | clasificador no lineal pequeno sobre el vector latente |

## Arquitectura del MLP usado

El MLP usado en este primer run fue deliberadamente pequeno. La intencion no
era buscar la mejor arquitectura posible, sino tener un probe no lineal ligero
que sirviera como contraste frente al probe lineal.

La arquitectura exacta fue:

```text
StandardScaler()
MLPClassifier(
  hidden_layer_sizes=(64,),
  alpha=0.0001,
  batch_size="auto",
  max_iter=300,
  random_state=42,
  early_stopping=False
)
```

En terminos practicos, cada vector latente se estandariza primero con
`StandardScaler`. Despues pasa por un `MLPClassifier` de scikit-learn con una
unica capa oculta de 64 unidades. La salida es una clasificacion multiclase
sobre los niveles observados en train. La activacion y el optimizador son los
defaults de scikit-learn: `relu` y `adam`.

El run guardo `mlp_dropout=0.0`. En el codigo, el parametro `dropout` solo se
pasa al `MLPClassifier` si la version instalada de scikit-learn lo soporta. Por
tanto, este resultado debe entenderse como una primera prueba con un MLP muy
simple, no como una busqueda optimizada ni como la arquitectura definitiva de la
futura cabeza de `two_head_sft`.

## Resultados principales

La tabla siguiente resume los probes aprendidos, ordenados por accuracy en
validation.

| estrategia | probe | accuracy | balanced acc | macro-F1 | weighted-F1 | MAE |
|---|---|---:|---:|---:|---:|---:|
| `record_prefix_state` | MLP | 0.8594 | 0.6400 | 0.6674 | 0.8506 | 0.2478 |
| `record_prefix_state` | linear | 0.8583 | 0.7647 | 0.7303 | 0.8514 | 0.2556 |
| `line_prefix_state` | MLP | 0.8072 | 0.7708 | 0.7364 | 0.8054 | 0.3806 |
| `line_prefix_state` | linear | 0.7983 | 0.7703 | 0.7365 | 0.7936 | 0.3733 |
| `line_first_token` | MLP | 0.7906 | 0.7212 | 0.6938 | 0.7846 | 0.4372 |
| `line_last_token` | MLP | 0.7894 | 0.6099 | 0.6191 | 0.7780 | 0.4228 |
| `line_mean` | linear | 0.7889 | 0.6387 | 0.6382 | 0.7805 | 0.4022 |
| `line_first_token` | linear | 0.7883 | 0.7621 | 0.7167 | 0.7832 | 0.4361 |
| `line_last_token` | linear | 0.7878 | 0.6236 | 0.6306 | 0.7775 | 0.4056 |
| `line_mean` | MLP | 0.7667 | 0.6276 | 0.6236 | 0.7578 | 0.4633 |

Los baselines quedaron claramente por debajo:

| baseline | accuracy | balanced acc | macro-F1 | weighted-F1 | MAE |
|---|---:|---:|---:|---:|---:|
| `previous_level` | 0.3522 | 0.3400 | 0.3455 | 0.3586 | 0.7611 |
| `majority` | 0.1633 | 0.1111 | 0.0312 | 0.0459 | 1.8578 |

La primera conclusion factual es que todos los probes aprendidos superan con
margen a los baselines. Esto indica que el resultado no se explica solo por la
clase mayoritaria ni por la continuidad local del nivel anterior. Hay una senal
recuperable en los hidden states del modelo base.

El mejor resultado por accuracy y MAE es `record_prefix_state + MLP`. Sin
embargo, esa no es toda la historia. `record_prefix_state + linear` obtiene una
accuracy practicamente igual, pero mejora bastante en `balanced_accuracy` y
`macro-F1`. Esto sugiere que el MLP optimiza muy bien el rendimiento global,
pero el probe lineal mantiene una distribucion mas equilibrada entre clases.

Por otro lado, `line_prefix_state` es la alternativa mas competitiva si se miran
metricas menos dominadas por las clases frecuentes. Tanto su version lineal como
su MLP alcanzan los mejores valores de `balanced_accuracy` y `macro-F1`, aunque
su accuracy global sea menor que la de `record_prefix_state`.

## Comparacion entre `record_prefix_state` y `line_prefix_state`

A primera vista, puede parecer sorprendente que `record_prefix_state` sea tan
fuerte. Es la estrategia mas estricta: no ve el `document_index`, no ve el
`line_index` y tampoco ve el contenido de la linea que se va a clasificar. Solo
dispone del prompt y de las lineas anteriores. Precisamente por eso su resultado
es interesante. El modelo base parece mantener en el estado autoregresivo una
representacion bastante informativa del punto estructural en el que se encuentra
el documento.

En YAML, esto tiene sentido. Tras observar una linea como `metadata:` o
`containers:`, el siguiente nivel no es arbitrario. El contexto anterior impone
expectativas muy fuertes sobre si la siguiente linea deberia abrir una entrada
anidada, continuar una lista o volver a un nivel anterior. `record_prefix_state`
parece capturar esa dinamica de "estado estructural" con suficiente calidad
como para predecir muchos niveles sin observar la linea actual.

`line_prefix_state`, por su parte, ve informacion adicional: el indice del
documento y la posicion de la linea. No ve todavia `line_text`, pero si conoce
que linea del documento esta a punto de generarse. Esta informacion puede ser
util, porque en este dataset las posiciones tempranas suelen corresponder a
campos superficiales como `apiVersion`, `kind`, `metadata` o `spec`, mientras
que las posiciones posteriores contienen con mas frecuencia estructuras
anidadas. Sin embargo, esa ventaja tambien abre la puerta a un posible sesgo
posicional: el probe podria aprender regularidades del dataset asociadas a
`line_index` en vez de una representacion jerarquica mas general.

## Analisis de matrices de confusion

La siguiente tabla compara el recall por nivel para las dos estrategias mas
importantes. Se incluyen las versiones lineal y MLP porque la diferencia entre
ambas ayuda a separar rendimiento global y equilibrio entre clases.

| level | support | record MLP | record linear | line-prefix MLP | line-prefix linear |
|---:|---:|---:|---:|---:|---:|
| 0 | 324 | 0.9938 | 0.9969 | 1.0000 | 0.9969 |
| 1 | 294 | 0.9422 | 0.9320 | 0.9456 | 0.9490 |
| 2 | 230 | 0.9652 | 0.9478 | 0.9739 | 0.9565 |
| 3 | 328 | 0.8689 | 0.8841 | 0.8262 | 0.8140 |
| 4 | 450 | 0.8333 | 0.8244 | 0.5978 | 0.5911 |
| 5 | 114 | 0.3684 | 0.3421 | 0.4123 | 0.3596 |
| 6 | 44 | 0.4545 | 0.4545 | 0.6818 | 0.6818 |
| 7 | 4 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 | 12 | 0.3333 | 0.5000 | 0.5000 | 0.5833 |

La diferencia mas importante aparece en `level=4`. Es la clase con mas soporte
en validation, con 450 lineas. `record_prefix_state + MLP` acierta 375 de esas
lineas, mientras que `line_prefix_state + MLP` acierta 269. Esta diferencia
explica gran parte de la ventaja de `record_prefix_state` en accuracy global.

En cambio, `line_prefix_state` mejora en varios niveles profundos. En `level=6`
pasa de 20 aciertos con `record_prefix_state + MLP` a 30 aciertos. Tambien
mejora en `level=7` y `level=8`, aunque aqui el soporte es muy pequeno y la
lectura debe ser prudente. El resultado no basta para afirmar que
`line_prefix_state` sea mejor para todos los casos profundos, pero si sugiere
que la posicion explicita de la linea puede aportar informacion util en ciertos
patrones de anidamiento.

El patron de errores para `level=4` es especialmente revelador. Con
`record_prefix_state + MLP`, los errores principales son:

```text
4 -> 4: 375
4 -> 2: 45
4 -> 3: 17
4 -> 5: 7
4 -> 6: 3
```

Con `line_prefix_state + MLP`, el mismo nivel se comporta asi:

```text
4 -> 4: 269
4 -> 2: 164
4 -> 3: 3
4 -> 5: 1
4 -> 6: 7
4 -> 7: 4
```

El problema de `line_prefix_state` no es que empuje sistematicamente estos
casos hacia niveles mas profundos, sino que colapsa muchas lineas de `level=4`
hacia `level=2`.

## Discusion sobre `line_index`

La preocupacion por `line_index` es metodologicamente relevante. En validation,
la posicion de la linea esta claramente correlacionada con el nivel medio:

| rango de `line_index` | lineas | nivel medio real |
|---|---:|---:|
| 0-4 | 406 | 0.22 |
| 5-9 | 374 | 1.68 |
| 10-14 | 366 | 2.80 |
| 15-19 | 342 | 4.16 |
| 20+ | 312 | 4.25 |

Esto significa que un probe que vea `line_index` podria explotar una regularidad
posicional del dataset. La pregunta es si el fallo de `line_prefix_state`
consiste simplemente en asumir que indices altos implican niveles altos. Al
analizar los errores por rango, no parece ser esa la explicacion principal.

| estrategia | rango | accuracy | nivel real medio | nivel predicho medio | error medio | MAE | over | under |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| line-prefix MLP | 0-4 | 0.9951 | 0.22 | 0.23 | +0.01 | 0.01 | 2 | 0 |
| line-prefix MLP | 5-9 | 0.9947 | 1.68 | 1.69 | +0.01 | 0.01 | 2 | 0 |
| line-prefix MLP | 10-14 | 0.9481 | 2.80 | 2.79 | -0.01 | 0.07 | 10 | 9 |
| line-prefix MLP | 15-19 | 0.5526 | 4.16 | 3.57 | -0.60 | 0.90 | 28 | 125 |
| line-prefix MLP | 20+ | 0.4519 | 4.25 | 3.20 | -1.05 | 1.11 | 9 | 162 |

En las lineas tardias, `line_prefix_state` no tiende a predecir niveles
demasiado altos. Ocurre lo contrario: infraestima. En el rango `20+`, el nivel
real medio es 4.25, pero el nivel predicho medio baja a 3.20. Ademas, hay 162
errores por debajo del nivel correcto y solo 9 por encima.

Comparado con `record_prefix_state`, el mismo patron existe pero es menos
acusado:

| estrategia | rango | accuracy | nivel real medio | nivel predicho medio | error medio | MAE | over | under |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| record MLP | 0-4 | 0.9877 | 0.22 | 0.23 | +0.00 | 0.01 | 3 | 2 |
| record MLP | 5-9 | 0.9973 | 1.68 | 1.69 | +0.01 | 0.01 | 1 | 0 |
| record MLP | 10-14 | 0.9399 | 2.80 | 2.83 | +0.02 | 0.06 | 15 | 7 |
| record MLP | 15-19 | 0.7135 | 4.16 | 3.69 | -0.47 | 0.54 | 12 | 86 |
| record MLP | 20+ | 0.5929 | 4.25 | 3.69 | -0.56 | 0.74 | 21 | 106 |

Por tanto, la hipotesis mas razonable no es que `line_prefix_state` aprenda una
regla monotona simple del tipo "cuanto mayor es `line_index`, mayor es
`level`". Parece mas bien que `line_index` aporta una pista fuerte al principio
y en posiciones medias, pero en posiciones tardias no basta para distinguir
entre continuaciones profundas, cierres parciales y campos que vuelven a niveles
intermedios. Cuando no ve el contenido de la linea, el probe tiende a resolver
esa ambiguedad desplazando muchas predicciones hacia niveles demasiado bajos.

## Fallos especificos de `record_prefix_state`

Aunque `record_prefix_state` sea el resultado mas fuerte globalmente, sus fallos
son muy informativos. La estrategia no ve nada de la linea actual, asi que su
prediccion depende por completo del estado previo del documento. Esto le da una
ventaja cuando el contexto anterior determina claramente la siguiente
profundidad, pero le limita cuando el siguiente paso depende de una decision
local que todavia no se ha expresado en texto.

En los niveles profundos, `record_prefix_state + MLP` tiende tambien a
infraestimar:

```text
5 -> 5: 42
5 -> 3: 37
5 -> 4: 26
5 -> 2: 5
```

```text
6 -> 6: 20
6 -> 4: 16
6 -> 3: 7
```

```text
8 -> 8: 4
8 -> 5: 7
8 -> 6: 1
```

Y en `level=7`, donde solo hay 4 ejemplos, el MLP predice todos como `level=3`.
No conviene sobreinterpretar ese caso por el soporte tan bajo, pero si encaja
con el patron general: cuando la profundidad real es rara o depende de una
continuacion muy especifica, el estado previo tiende a proyectarla hacia un
nivel intermedio mas comun.

La comparacion con `line_prefix_state` permite matizar el resultado. `record`
es mas estable en niveles frecuentes y en posiciones tardias; por eso gana en
accuracy y MAE. `line_prefix`, en cambio, detecta mejor algunos niveles
profundos concretos, pero pierde muchas lineas de `level=4` al predecirlas como
`level=2`. Dicho de otra manera: `record_prefix_state` parece capturar mejor el
estado estructural global, mientras que `line_prefix_state` incorpora una senal
posicional que a veces ayuda en profundidad, pero que no resuelve bien la
ambiguedad local de las lineas tardias.

## Como mejorar el probe MLP

El MLP actual es util como primer contraste no lineal, pero no deberia tomarse
como la mejor arquitectura posible. Hay varias mejoras razonables antes de usar
sus resultados como guia fina para `two_head_sft`.

La primera mejora seria hacer una busqueda pequena y controlada de
hiperparametros. Una matriz suficiente para el siguiente paso podria probar
dimensiones ocultas `64`, `128` y `256`; una o dos capas; dropout `0.0`, `0.1`
y `0.2`; y valores de regularizacion `alpha` en `1e-5`, `1e-4` y `1e-3`. Esta
busqueda no deberia optimizar solo accuracy, sino tambien `balanced_accuracy`,
`macro-F1`, MAE y errores por rangos de profundidad.

La segunda mejora seria introducir early stopping con una particion interna de
train. El run actual usa `early_stopping=False`, por lo que no hay una curva de
validacion interna que permita distinguir si el MLP esta aprovechando una senal
real o sobreajustando regularidades concretas del split.

La tercera mejora seria implementar el probe en PyTorch. Esto permitiria
controlar mejor dropout, guardar curvas de entrenamiento, usar ponderacion por
clase y experimentar con perdidas mas adecuadas para una variable ordinal. El
`level` no es una clase nominal cualquiera: equivocarse de 4 a 5 no es igual
que equivocarse de 4 a 1. Por eso tiene sentido probar una perdida combinada de
cross-entropy y penalizacion por distancia, o una formulacion de regresion
ordinal.

Por ultimo, hay que separar mejora del MLP y control de leakage o sesgo. Antes
de interpretar una arquitectura mas potente como evidencia fuerte, conviene
anadir dos ablations: un baseline que use solo `document_index` y `line_index`,
y una variante de `line_prefix_state` sin `line_index`. Si el baseline
metadata-only ya obtiene un resultado alto, parte de la senal vendria de la
estructura posicional del dataset. Si `line_prefix_state` sin `line_index`
mantiene buen rendimiento, la evidencia a favor de una representacion
estructural interna seria mas solida.

## Conclusiones

El resultado principal del run es que el modelo base contiene una senal
recuperable sobre el `level` jerarquico de las lineas YAML. Los probes aprendidos
superan ampliamente a los baselines, y los probes lineales ya son competitivos.
Esto sugiere que la informacion no solo existe, sino que en parte esta
organizada de forma bastante accesible.

El resultado mas llamativo es la fuerza de `record_prefix_state`. Al no ver ni
la posicion explicita ni el contenido de la linea actual, su buen rendimiento
apunta a que el contexto autoregresivo previo ya codifica un estado estructural
util. Esta observacion es especialmente relevante para `two_head_sft`, porque
abre la posibilidad de predecir el nivel de una linea antes de generar su
contenido, al menos en muchos casos.

Al mismo tiempo, `line_prefix_state` no debe descartarse. Aunque pierde en
accuracy global, parece mas sensible a algunos niveles profundos. La lectura
prudente es que esta estrategia combina senal estructural y senal posicional:
puede ayudar, pero tambien puede introducir sesgos por `line_index`. Por eso el
siguiente paso no deberia ser elegir una estrategia definitiva, sino ejecutar
ablations que separen contexto previo, posicion explicita y contenido de la
linea.

En resumen, este probe no demuestra todavia que `two_head_sft` vaya a mejorar
la generacion final de manifiestos Kubernetes. Pero si aporta una evidencia
diagnostica importante: el modelo base ya representa parte de la jerarquia YAML
en sus hidden states, y esa senal es suficientemente fuerte como para justificar
seguir desarrollando una cabeza explicita de nivel y analizar con mas detalle
donde debe conectarse.
