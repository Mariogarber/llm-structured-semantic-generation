# Notas para memoria: problemas observados en `two_head_sft`

## Contexto

El experimento `two_head_sft` introduce una separacion explicita entre dos
tareas que en el SFT serializado normal aparecen mezcladas en la misma
superficie textual. Por un lado, el modelo base genera el contenido visible de
cada linea en formato `content_blocks_v1`. Por otro lado, una cabeza estructural
predice el `level` de cada linea a partir de hidden states del transformer.

Esta decision responde directamente a la pregunta principal del proyecto: si la
jerarquia de un manifiesto YAML debe aprenderse como texto serializado o como
una senal estructural separada. Sin embargo, los resultados del primer run
muestran un problema importante: muchas predicciones de `two_head_sft` no llegan
a YAML parseable. A primera vista esto hace que las metricas globales parezcan
mucho peores que las del SFT normal. El analisis posterior matiza esa lectura:
cuando se evalua solo sobre los ejemplos parseables, ambos modelos son bastante
parecidos en varias metricas, e incluso `two_head_sft` mantiene un `level_mae`
competitivo.

Por tanto, el problema principal observado no es simplemente que el modelo
genere contenido inutil. El problema es que una fraccion alta de salidas cae
antes de llegar a la zona donde las metricas semanticas y estructurales se
pueden medir con normalidad. Esto convierte la parseabilidad en el cuello de
botella de la arquitectura.

## Efecto sobre las metricas

El evaluador esta construido de forma conservadora: si una prediccion no puede
reconstruirse como YAML parseable o no cumple el contrato parser-facing de
bloques, muchas metricas posteriores se asignan a cero. Esto afecta a metricas
como `line_text_f1`, `content_exact_match_rate`, `level_exact_match_rate`,
`semantic_key_f1`, `prompt_requirement_f1` y `kind_sequence_match_rate`.

Esta decision es razonable desde el punto de vista de evaluacion estructural,
porque una salida no parseable no es util como manifiesto Kubernetes. Sin
embargo, tambien hace que las metricas globales mezclen dos fenomenos distintos:
la calidad del contenido generado y la capacidad de la arquitectura para
mantener una superficie parseable. En el primer run de `two_head_sft`, la caida
global esta dominada por el segundo fenomeno.

La comparacion parseable-only es importante precisamente por eso. En los casos
en los que `two_head_sft` si produce una salida parseable, las diferencias con
`serialized_sft` son mucho menores. Esto sugiere que la arquitectura no falla de
forma uniforme en toda la tarea, sino que tiene un modo de fallo especifico:
mantener la coherencia estructural necesaria para que el parser pueda reconstruir
YAML valido.

## Problemas de superficie corregibles sin reentrenar

Se han identificado dos fallos que no dependen directamente de la capacidad del
modelo para entender Kubernetes ni de la cabeza estructural de nivel. Ambos
pertenecen a la serializacion auxiliar `content_blocks_v1`.

El primero aparece cuando el modelo genera una fila con dos campos en vez de
tres. El contrato visible esperado es:

```text
document_index    line_index    line_text
```

Pero algunas salidas tienen esta forma:

```text
line_index    line_text
```

En esos casos, si ya se ha leido una fila valida dentro del bloque, el
`document_index` puede recuperarse de forma determinista usando el documento
anterior. Este arreglo no introduce contenido nuevo, no consulta la referencia
gold y no modifica el `level`; solo recupera una variable posicional que la
propia secuencia ya hace evidente.

El segundo fallo aparece cuando `line_index` no es consecutivo, se repite o
retrocede. En `content_blocks_v1`, el orden real de las lineas ya viene dado por
el orden de generacion. Por tanto, el `line_index` visible funciona como una
ayuda de superficie, no como el contenido estructural que se quiere evaluar. Se
ha aplicado una normalizacion que renumera las lineas por orden dentro de cada
`document_index` antes de pasar a la prediccion de niveles y a la reconstruccion
parser-facing.

Estos dos arreglos deben describirse como normalizacion de superficie, no como
mejora del modelo. Su objetivo es evitar que el sistema falle por variables
posicionales recuperables, manteniendo intactos los problemas realmente
estructurales: el texto de la linea y el nivel predicho.

## Problemas no corregibles de forma limpia por postproceso

Otros fallos observados no pueden arreglarse sin cambiar el comportamiento del
modelo o sin introducir reparaciones que alterarian el significado de la salida.
El caso mas importante es la mala reconstruccion de listas y mapas anidados,
especialmente en recursos como `DaemonSet`, donde aparecen bloques `containers`,
`command`, `volumeMounts` o `volumes`. Estos fragmentos dependen de niveles
profundos y de una relacion fina entre claves, listas y valores. Si el `level`
predicho es demasiado bajo, o si una linea que deberia estar dentro de una lista
queda al mismo nivel que su padre, PyYAML produce errores como `expected <block
end>, but found '-'` o `mapping values are not allowed here`.

Este tipo de error no se puede corregir de forma segura solo con reglas locales.
Mover una linea a un nivel mas profundo puede hacer que el YAML parse, pero
tambien puede cambiar el objeto Kubernetes resultante. En este proyecto el
parser actua como control estructural, no como mecanismo oculto de reparacion.
Por eso no conviene introducir heuristicas que "adivinen" donde deberia ir cada
linea si el modelo no lo ha predicho.

Tambien aparecen fragmentos de comandos shell divididos en varias lineas YAML.
En algunos ejemplos, el modelo genera partes de un comando como si fueran lineas
independientes del manifiesto. Esto puede romper el parser incluso cuando la
intencion semantica es reconocible. Aun asi, reconstruir automaticamente esos
fragmentos requeriria decidir si una linea pertenece a un scalar, a una lista de
comandos o a un campo YAML ordinario. Esa decision ya no es una normalizacion
posicional; es una reparacion semantica y, por tanto, debe tratarse con cautela.

## Problema de niveles profundos

El hallazgo mas relevante para la arquitectura es que la cabeza estructural del
primer run no predice niveles por encima de 4, aunque la validacion contiene
niveles 5, 6, 7 y 8. Este patron encaja parcialmente con el analisis previo de
hidden states: los niveles profundos eran precisamente las clases mas dificiles
para el probe, sobre todo con `record_prefix_state`. Sin embargo, el colapso
total a niveles `0..4` es mas severo que lo que sugeria el probe.

Este punto es importante para la memoria porque conecta el resultado empirico
con la hipotesis de representacion. `record_prefix_state` observa el estado justo
antes de cada registro de contenido. Es una posicion limpia para alinear una
etiqueta por linea, pero puede no contener suficiente informacion sobre la
estructura fina que aparece despues en `line_text`, especialmente en listas
profundas y fragmentos anidados. En el analisis de probes, otros candidatos como
`line_prefix_state` mostraban mejor comportamiento relativo en niveles
profundos, aunque con peores compromisos globales.

Por tanto, el resultado no invalida la idea de una cabeza estructural, pero si
senala que la eleccion del hidden state es critica. La comparacion futura deberia
separar dos preguntas: si predecir `level` con una cabeza explicita es util, y
que estado del transformer proporciona la informacion adecuada para esa cabeza.

## Relacion con `serialized_sft`

El SFT serializado normal tiene una ventaja practica clara: aprende una unica
superficie `blocks_tsv_v1` donde `line_text` y `level` aparecen juntos como
texto. Esto hace que la generacion pueda capturar correlaciones locales entre
una linea y su nivel dentro de la misma secuencia autoregresiva. El coste es que
la jerarquia se trata como texto ordinario.

`two_head_sft`, en cambio, fuerza una separacion mas interesante desde el punto
de vista de investigacion: el contenido se genera como texto y la jerarquia se
predice como senal estructural. Esa separacion permite estudiar de forma mas
directa si el modelo contiene informacion util sobre la jerarquia en sus hidden
states. Pero tambien introduce una dependencia adicional: si la cabeza de nivel
no recupera bien los niveles profundos, el contenido puede ser razonable y aun
asi producir YAML no parseable.

La lectura mas justa de los resultados actuales es, por tanto, doble. Como
sistema final, `two_head_sft` queda limitado por parseabilidad. Como experimento
comparativo, ofrece una senal valiosa: cuando la salida parsea, el rendimiento
no esta tan lejos del SFT normal; cuando falla, los errores se concentran en la
interfaz entre contenido generado, nivel predicho y parser.

## Cambios aplicados despues del primer run

Despues de la auditoria se aplicaron dos normalizaciones en
`scripts/train_kubernetes_two_head_sft.py`:

1. Recuperacion de `document_index` en filas de dos campos cuando el documento
   activo es inequivocamente el anterior.
2. Renumeracion determinista de `line_index` por orden de aparicion dentro de
   cada `document_index`.

Ademas, el extractor de `content_blocks_v1` deja de truncar silenciosamente
salidas malformadas despues de filas validas. Si aparece una fila que no puede
normalizarse de forma segura, la validacion debe registrarla como error de
superficie. Esto hace que la metrica de salida estructurada sea menos optimista
y mas fiel al contrato real.

Estos cambios no modifican las metricas originales documentadas para el run
`two-head-sft-v1-20260516`. Si se recomputa la validacion despues de estos
arreglos, ese resultado debe etiquetarse como evaluacion postprocesada o
normalizada, no como el resultado directo del entrenamiento.

## Implicacion metodologica

Para la memoria, este episodio puede presentarse como una distincion necesaria
entre tres niveles de fallo. El primer nivel es la serializacion auxiliar, donde
algunos errores son recuperables porque afectan a campos posicionales. El
segundo nivel es la reconstruccion YAML, donde errores de jerarquia hacen que el
parser rechace la salida. El tercer nivel es la calidad semantica Kubernetes,
que solo puede evaluarse de forma fiable una vez superados los dos anteriores.

Esta separacion evita una conclusion demasiado simple. No basta con decir que
`two_head_sft` es peor porque tiene menor parseability, ni tampoco basta con
ignorar la parseability porque en los casos parseables se parece al SFT normal.
Lo relevante es que la arquitectura separa contenido y jerarquia, y precisamente
esa frontera se convierte en el punto fragil del primer experimento.
