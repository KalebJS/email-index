from django.db import models


class Email(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    subject = models.CharField(max_length=255)
    sender = models.EmailField()
    date = models.DateTimeField()
    body = models.TextField()
    html = models.TextField()
    filename = models.CharField(max_length=255)
