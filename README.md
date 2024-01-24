# Descripción del proyecto
Trabajamos como analistas para el operador de telecomunicaciones Megaline.   
La empresa ofrece a sus clientes dos tarifas de prepago, Surf y Ultimate.   
El departamento comercial quiere saber cuál de los planes genera más ingresos para poder ajustar el presupuesto de publicidad.  
Vamos a realizar un análisis preliminar de las tarifas basado en una selección de clientes relativamente pequeña.   
Tendemos los datos de 500 clientes de Megaline: quiénes son los clientes, de dónde son, qué tarifa usan, así como la cantidad de llamadas que hicieron y los mensajes de texto que enviaron en 2018.   
Nuestro trabajo es analizar el comportamiento de los clientes y determinar qué tarifa de prepago genera más ingresos. 

# Descripción de las tarifas

A continuación puedes ver una descripción de las tarifas:
1. Surf
- Pago mensual: 20$.
- 500 minutos al mes, 50 SMS y 15 GB de datos.
## Si se exceden los límites del paquete:
- 1 minuto: 3 centavos.
- 1 SMS: 3 centavos.
- 1 GB de datos: 10$.
2. Ultimate
- Pago mensual: 70$.
- 3000 minutos al mes, 1000 SMS y 30 GB de datos.
## Si se exceden los límites del paquete:
- 1 minuto: 1 centavo.
- 1 SMS: 1 centavo.
- 1 GB de datos: 7$.
# Diccionario de datos
En este proyecto, trabajaremoss con cinco tablas diferentes.
1. La tabla users (datos sobre los usuarios):
 
- user_id: identificador único del usuario.
- first_name: nombre del usuario.
- last_name: apellido del usuario.
- age: edad del usuario (en años).
- reg_date: fecha de suscripción (dd, mm, aa).
- churn_date: la fecha en la que el usuario dejó de usar el servicio (si el valor es ausente, la tarifa se estaba usando cuando fue extraída esta base de datos).
- city: ciudad de residencia del usuario.
- plan: nombre de la tarifa.
2. La tabla calls (datos sobre las llamadas):
 
- id: identificador único de la llamada.
- call_date: fecha de la llamada.
- duration: duración de la llamada (en minutos).
- user_id: el identificador del usuario que realiza la llamada.
3. La tabla messages (datos sobre los SMS):
 
- id: identificador único del SMS.
- message_date: fecha del SMS.
- user_id: el identificador del usuario que manda el SMS.
4. La tabla internet (datos sobre las sesiones web):
 
- id: identificador único de la sesión.
- mb_used: el volumen de datos gastados durante la sesión (en megabytes).
- session_date: fecha de la sesión web.
- user_id: identificador del usuario.
5. La tabla plans (datos sobre las tarifas):
 
- plan_name: nombre de la tarifa.
- usd_monthly_fee: pago mensual en dólares estadounidenses.
- minutes_included: minutos incluidos al mes.
- messages_included: SMS incluidos al mes.
- mb_per_month_included: datos incluidos al mes (en megabytes).
- usd_per_minute: precio por minuto tras exceder los límites del paquete (por ejemplo, si el paquete incluye 100 minutos, el operador cobrará el minuto 101).
- usd_per_message: precio por SMS tras exceder los límites del paquete.
- usd_per_gb: precio por gigabyte de los datos extra tras exceder los límites del paquete (1 GB = 1024 megabytes).