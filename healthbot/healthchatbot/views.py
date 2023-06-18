from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from healthchatbot.chat import get_response
import os
import traceback
import json

@api_view(['POST'])
def user_conversation(request):
    try:
        input = request.data['message']
        result = get_response(input)
        output = {"answer": result}
        return Response(output, 200)
    except:
        traceback.print_exc()
        result = {"answer" : "error"}
        return Response(result, 500)
