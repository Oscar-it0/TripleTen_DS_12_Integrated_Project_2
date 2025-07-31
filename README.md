#  Sprint 12 - Proyecto de Recuperación de Oro

Este proyecto tiene como objetivo construir un modelo de machine learning para predecir la recuperación de oro en una planta de procesamiento. Utilizamos datos reales de producción, que incluyen parámetros físicos y químicos medidos en diferentes etapas del proceso.

##  Archivos de Datos

Los datos se encuentran en tres archivos CSV:
- `gold_recovery_train.csv`: conjunto de entrenamiento.
- `gold_recovery_test.csv`: conjunto de prueba (sin objetivos).
- `gold_recovery_full.csv`: conjunto completo con todas las características.

> Los datos están indexados por fecha y hora (`date`). Los parámetros cercanos en el tiempo suelen ser similares.

##  Instrucciones del Proyecto

### 1. Preparación de los Datos
-  Cargar y explorar los archivos desde:
  `/datasets/gold_recovery_train.csv`
  `/datasets/gold_recovery_test.csv`
  `/datasets/gold_recovery_full.csv`
-  Verificar el cálculo de `rougher.output.recovery` y comparar con los valores reales usando el Error Absoluto Medio (EAM).
-  Identificar las características ausentes en el conjunto de prueba y analizar su tipo.
-  Realizar el preprocesamiento necesario (valores nulos, tipos de datos, etc.).

### 2. Análisis Exploratorio
-  Analizar cómo cambian las concentraciones de **Au**, **Ag** y **Pb** en cada etapa del proceso.
-  Comparar la distribución del tamaño de partículas entre los conjuntos de entrenamiento y prueba.
-  Evaluar la suma total de concentraciones en cada etapa para detectar valores anómalos y decidir si deben eliminarse.

### 3. Construcción del Modelo
-  Implementar una función para calcular el **sMAPE** (Symmetric Mean Absolute Percentage Error).
-  Entrenar múltiples modelos y evaluarlos con validación cruzada.
-  Seleccionar el mejor modelo y probarlo con el conjunto de prueba.

##  Criterios de Evaluación
Los revisores evaluarán:
- Calidad del análisis y preparación de datos.
- Variedad y rendimiento de los modelos desarrollados.
- Correcta validación y evaluación del modelo.
- Claridad en la explicación de cada paso.
- Limpieza y organización del código.
- Conclusiones obtenidas.