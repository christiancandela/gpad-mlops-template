# Datos y gobernanza (Entrega 2)

## 1. Dataset
Fuente · licencia · tamaño · granularidad · diccionario de variables · variable objetivo.

## 2. Diagnóstico de calidad
| Dimensión | Regla verificada | Resultado | Acción |
|---|---|---|---|
| Completitud | | | |
| Validez | | | |
| Consistencia | | | |
| Unicidad | | | |
| Actualidad | | | |

## 3. Hallazgos del EDA

## 4. Contrato de datos
Implementado en `contracts/churn_contract.py` y verificado en la etapa `validate` de `dvc.yaml`.

| Campo | Tipo | Reglas | Responsable | Acción si falla |
|---|---|---|---|---|
| tenure_months | int | 0–72 | Dueño del CRM | Detener pipeline |
| contract_type | str | {month-to-month, one-year, two-year} | Dueño del CRM | Detener pipeline |
| … | | | | |

## 5. Plan de gobernanza
| Elemento | Decisión |
|---|---|
| Propietario y custodio del dato | |
| Política de acceso | |
| Datos personales y tratamiento (Ley 1581 de 2012) | |
| Versionamiento (DVC y remoto) | |
| Linaje (crudo → validado → train/test → modelo) | `dvc dag` |
| Retención | |
