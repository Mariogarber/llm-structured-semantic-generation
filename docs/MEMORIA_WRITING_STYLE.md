# Memoria Writing Style

Este documento fija la guía de estilo que debe seguirse al redactar o revisar la memoria del proyecto. Su objetivo no es imponer una plantilla cerrada, sino conservar una voz reconocible y coherente durante capítulos largos, especialmente cuando participen distintos agentes o sesiones.

## Voz General

El estilo debe ser argumentativo y progresivo. La escritura no debe lanzar la tesis de golpe, sino construir una cadena de razonamiento: primero una intuición práctica, después una limitación, luego el caso concreto y finalmente la pregunta técnica o metodológica. En este sentido, el texto debe dar la sensación de que el problema se está delimitando con cuidado, no simplemente declarando.

La voz debe ser académica, pero cercana. La memoria puede usar términos técnicos como generación autoregresiva, estructura de árbol, serialización YAML, cabezal adicional o nivel jerárquico, pero estos términos deben convivir con lenguaje natural y transiciones humanas. Expresiones como "a primera vista", "sin embargo", "precisamente por eso" o "por otro lado" son útiles cuando ayudan a mostrar el razonamiento.

## Forma Del Razonamiento

La explicación debe apoyarse con frecuencia en contrastes: texto libre frente a estructura formal, secuencia plana frente a jerarquía, comprensión del prompt frente a serialización correcta, o solución aparentemente sencilla frente a problema estructural más profundo. Estos contrastes son importantes porque encajan con la pregunta central del proyecto: hasta qué punto un modelo que genera tokens en secuencia puede producir una salida cuya validez depende de una organización jerárquica.

Los párrafos pueden ser medianos o largos, siempre que tengan una idea central clara. Normalmente deben empezar con una afirmación general y desarrollarla con matices, consecuencias y una transición hacia el siguiente punto. Deben evitarse los fragmentos telegráficos salvo en esquemas, notas internas o listas de trabajo.

## Relación Con El Proyecto

Al redactar la memoria, debe conservarse el marco actual del repositorio: generación estructurada de manifiestos Kubernetes, comparación entre `serialized_sft` y `two_head_sft`, uso del `level` como variable estructural explícita, parser como control estructural y evaluación centrada en validez sintáctica, estructural y semántica aproximada. El texto no debe volver a una formulación genérica de generación YAML ni presentar el parser como un sistema de reparación semántica.

También debe mantenerse un tono exploratorio, no dogmático. Las decisiones del proyecto pueden justificarse con claridad, pero no deben presentarse como resultados demostrados si todavía son hipótesis, ramas documentadas o componentes en desarrollo.

## Patrones A Evitar

Debe evitarse cualquier patrón típico de respuesta genérica de IA: enumeraciones mecánicas, cierres demasiado simétricos, frases promocionales, afirmaciones grandilocuentes o párrafos que parezcan independientes entre sí. La memoria debe parecer pensada en voz alta, pero luego pulida para un contexto académico.

Antes de entregar una sección, revisar que cada párrafo avance el argumento, que las transiciones sean explícitas y que el texto combine precisión técnica con una presencia humana reconocible.
