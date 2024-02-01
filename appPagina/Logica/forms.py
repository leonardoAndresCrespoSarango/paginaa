from django import forms
from appPagina.Logica.modeloSNN import ImageModel


class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = ImageModel
        fields = ['image']