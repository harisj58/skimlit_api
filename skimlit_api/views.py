from django.http import JsonResponse
from rest_framework.decorators import api_view
import tensorflow as tf
from keras.layers import TextVectorization
import pandas as pd
import json

model_path = "skimlit_api/skimlit_tribrid_model/"

# Load saved model
loaded_model = tf.keras.models.load_model(model_path)

@api_view(['GET'])
def get_skim(request):
    if request.method == 'GET':
        abs_text = request.GET.get('abstract')
        labels = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]
        sent_list = abs_text.split(sep=".")
        sent_list.pop()
        sent_list = [sentence.strip() for sentence in sent_list]
        i = 0
        total_lines = len(sent_list)
        final_list = []
        temp = {}
        for line in sent_list:
            temp["text"] = line
            temp["line_number"] = i
            temp["total_lines"] = total_lines
            i += 1
            final_list.append(temp)
            temp = {}
        df = pd.DataFrame(final_list)
        chars = [" ".join(list(sentence)) for sentence in sent_list]
        line_numbers_one_hot = tf.one_hot(df.line_number.to_numpy(), depth=15)
        lines_total_one_hot = tf.one_hot(df.total_lines.to_numpy(), depth=20)
        preds = tf.argmax(
            loaded_model.predict(
                x=(
                    line_numbers_one_hot,
                    lines_total_one_hot,
                    tf.constant(sent_list),
                    tf.constant(chars),
                ),
                verbose=0,
            ),
            axis=1,
        )
        result_dict = {}
        i = 0
        for i in range(0, total_lines):
            if i != 0 and preds[i] != preds[i - 1]:
                result_dict[labels[preds[i]]] = []
                result_dict[labels[preds[i]]].append(sent_list[i])
            elif i == 0:
                result_dict[labels[preds[i]]] = []
                result_dict[labels[preds[i]]].append(sent_list[i])
            else:
                result_dict[labels[preds[i]]].append(sent_list[i])
        return JsonResponse(result_dict, safe=False)
