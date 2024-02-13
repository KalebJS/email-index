from django.db import models


class Sender(models.Model):
    email = models.EmailField()
    name = models.CharField(max_length=255)


class Email(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    subject = models.CharField(max_length=255)
    date = models.DateTimeField()
    body = models.TextField()
    html = models.TextField()
    sender = models.ForeignKey(Sender, on_delete=models.CASCADE)
