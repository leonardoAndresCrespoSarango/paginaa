from django.contrib import admin
from django.urls import path
from appPagina.View import views
from django.conf import settings  
from django.conf.urls.static import static

urlpatterns = [
    path('', views.base, name='base'),  # base.html ahora es tu página principal
    path('deteccion/', views.deteccion, name='deteccion'),  # deteccion ahora tiene su propia URL
    # Otras rutas de tu aplicación
]

# Agregar estas líneas para servir archivos estáticos en modo de desarrollo
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
