"""Escenario de pruebas de carga con Locust para la API de inferencia de MiBici.

Entregable: Microservicios, CI/CD y FinOps
Responsable: Gaby
"""

# TODO (Gaby): Diseñar e implementar escenario de pruebas con Locust:
# 1. Definir clase de usuario y tiempos de espera entre peticiones
# 2. Configurar tareas con pesos diferenciados para endpoints del servicio:
#    - Health check / monitoreo sintético
#    - Inferencia unitaria (/predict) con datos de entrada representativos
#    - Inferencia en lote (/predict/batch)
# 3. Validar códigos de respuesta y tiempos de respuesta esperados
