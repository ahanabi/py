from django.urls import path
from .views import views
from .views import gauge

urlpatterns = [
    path('gauge/gauge_train', gauge.gauge_train, name='gauge_train'),
    path('gauge/gauge_predict', gauge.gauge_predict, name='gauge_predict'),
    path('gauge/gauge_save', gauge.gauge_save, name='gauge_save'),
    path('', views.index, name='index'),
]