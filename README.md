# MLOps Trabajo Final EAFIT

El trabajo final consiste de utilizar un algoritmo de clasificación con KNN para clasificar imágenes
como hombre o mujer.

## Configuración

Se necesita `pyenv` y `pipenv` para iniciar el proyecto.

```terminal
pyenv install 3.11
pip install pipenv
pipenv install
```

## Variables de entorno a utilizar

* `WANDB_API_KEY`: la API key para subir artefactos y métricas a WandB.

```terminal
cp .env.template .env
```

Dentro de `.env`, poner el valor de `WANDB_API_KEY`.

## Scripts importantes

> Antes de ejecutar los scripts, es necesario tener las variables de entorno configuradas. También es necesario estar dentro del ambiente virtual con `pipenv shell`

* `src/data/load.py`: Es el encargado de subir la data de entrenamiento y test a WandB como un artefacto. Se ejecuta con `python src/data/load.py --IdExecution=1 --no-dryRun`.
* `src/model/build.py`: Es el encargado de subir la definición y los parámetros del modelo a WandB. Se ejecuta con `python src/model/build.py --IdExecution=1 --no-dryRun`.
* `src/model/train.py --IdExecution=1 --no-dryRun`: Es el encargado de subir el modelo entrenado y subir las métricas a WandB. Se ejecuta con `python src/model/train.py`.
* `src/model/test.py --IdExecution=1 --no-dryRun`: Es el encargado de subir las métricas de test a WandB. Se ejecuta con `python src/model/test.py`.

> El `--IdExecution` es un identificador único para cada ejecución. Se recomienda usar el último número reportado en WandB y sumarle 1. Pero si no se desea hacer esto simplemente usar el número 1.

Si en vez de usar `--no-dryRun` se utilza `--dryRun`, ningún artefacto se subirá a WandB ni tampoco las métricas, pero
sí se correrá un job.
