from collections import defaultdict

import pandas as pd
from django.http import JsonResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rfp_api.models import Email

from .milvus_index import MilvusConnectionSecrets, MilvusService
from .html_utils import html_to_sentences

credentials = MilvusConnectionSecrets(user="username", password="password", host="standalone")
index = MilvusService(credentials, reset=True)


class Inference(APIView):
    def post(self, request) -> JsonResponse:
        payload = request.data
        if not (email := payload.get("query")):
            return JsonResponse({"error": "query is required"}, status=400)

        return_count = int(payload.get("count", 2))
        threshold = float(payload.get("threshold", 0.4))

        query_results = index.search(email, k=10, threshold=threshold)

        # get nearest neighbor
        classes = defaultdict(int)
        for item in query_results:
            classes[item.email_id] += item.score

        # create list of tuples (score, class) and sort it
        sorted_classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)
        email_ids = sorted_classes[:return_count]

        answers = []
        for email_id, score in email_ids:
            email = Email.objects.get(id=email_id)
            answers.append({"subject": email.subject, "similar_question": email.html, "score": score})

        return JsonResponse(answers, safe=False)


class EmailCreate(APIView):
    def post(self, request, format=None):
        try:
            email = Email.objects.create(**request.data)
        except Exception as e:
            return Response(str(e), status=status.HTTP_400_BAD_REQUEST)

        rows = []
        for sentence in html_to_sentences(email.html):
            rows.append({"text": sentence, "email_id": email.id})
        index.insert(pd.DataFrame(rows))
        return Response({"id": email.id}, status=status.HTTP_201_CREATED)
