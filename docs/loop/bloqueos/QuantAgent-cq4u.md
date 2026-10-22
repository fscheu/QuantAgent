# Bloqueo — QuantAgent-cq4u

Intenté recalcular `cartera-nucleo-curva` con el comando documentado en
`QuantAgent-ro3w-IM-criterio-paper-curva.md`. Las curvas están disponibles en
`/home/azureuser/repos/projects/QuantAgent/local-data/curvas-l14-curva`, pero el
comando base falla al importar VectorBT: Numba no puede crear su caché al leer
la instalación desde el checkout compartido. Con `NUMBA_CACHE_DIR=/tmp/numba-cache`
el cálculo corre, pero entrega DSR 0,26 (N efectivo 97), distinto del IM ro3w
(0,41; N efectivo 96), señal de que los registros cambiaron desde aquel informe.

El smoke también falla por el mismo error de caché. `pytest -m oraculo` quedó
sin salida completa en esta corrida, así que no puedo declarar verde el punto
de control. No escribí curvas ni registros.

Para destrabar: PM debe fijar qué snapshot de `intentos.csv` corresponde al
punto de control y correr el cálculo, el pedido en seco y los oráculos en la VM
con una configuración de Numba que funcione. Después se puede completar el IM
con las cuatro filas y comprobar que ninguna ventana nueva se abrió.
