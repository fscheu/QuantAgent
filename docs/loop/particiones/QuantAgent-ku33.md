# Partición QuantAgent-ku33: veredicto de cartera de F1

La entrega excede el objetivo de 100 líneas: corrige el cálculo C0 desde las curvas guardadas y agrega una recuperación CLI que integra M1, registro del intento, estado e idempotencia. Se divide en dos cambios verificables por separado.

1. **QuantAgent-ku33.1 — C0 desde curvas guardadas.** `validar_etapa3` debe leer los nombres que escribe el motor y juzgar la curva de cartera con los criterios de etapa 3 existentes. Binario: una prueba sintética con `{activo}.csv` comprueba C0 de cartera, DSR contra intentos y benchmark 1/N, sin simular la lectura de curvas. Revisión: probar.
2. **QuantAgent-ku33.2 — recuperación del veredicto.** Agregar un subcomando que consume la reserva ya guardada, calcula M1, registra intento y estado, y no duplica registros. Binario: una ejecución sintética verifica M1 frente al valor esperado; la repetición deja un intento y un estado; la prueba falla si se llama `abrir()` o `correr_motor`. Revisión: probar.

La segunda parte depende de la primera. Ninguna parte abre la reserva ni lee datos reales.
