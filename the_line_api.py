import os
import requests as remote_requests
import pandas as pd
import numpy as np
from fastapi import *
from typing import *
from pydantic import BaseModel
import time

##
from uuid import UUID, uuid4
from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters

from fastapi_session import *

class chat_input(BaseModel):
    messages: List[dict[str, str]]

############

cookie_params = CookieParameters()

# Uses UUID
cookie = SessionCookie(
    cookie_name="cookie",
    identifier="general_verifier",
    auto_error=True,
    secret_key="DONOTUSE",
    cookie_params=cookie_params,
)
backend = InMemoryBackend[UUID, SessionData]()

verifier = BasicVerifier(
    identifier="general_verifier",
    auto_error=True,
    backend=backend,
    auth_http_exception=HTTPException(status_code=403, detail="invalid session"),
)

############

system_prompt = f'You are a large language model named The Line Safety Chatbot developed by TONOMUS to answer general questions regarding construction safety, such as "What should be ensured before work commences?". Your response should be short and abstract, less than 64 words. Conversations should flow, and be designed in a way to not reach a dead end by ending responses with "Do you have any further questions?"'

app = FastAPI()


@app.post("/upload_qa_pairs/")
async def upload_qa_pair_excel_file(
    file: UploadFile,
    response: Response,
    ):

    ## upload the qa pairs
    out_file = open(
        os.path.join(
        'uploaded',
        file.filename
        ), 'wb+')
    
    out_file.write(file.file.read())
    out_file.close()

    ## read the qa pairs
    hr_training_pairs = pd.read_excel(
        os.path.join(
        'uploaded',
        file.filename
        ),
        )
    hr_training_pairs = hr_training_pairs[['Question', 'Answer']]
    hr_training_pairs = hr_training_pairs.drop_duplicates()
    hr_training_pairs = hr_training_pairs.to_dict('records')

    ## embedding
    qa_pairs = []

    start_time = time.time()
    for r in hr_training_pairs:
        try:

            Question_embedding = remote_requests.post(
                'http://37.224.68.132:27329/text_embedding/all_mpnet_base_v2',
                json = {
                  "text": r['Question']
                }
                ).json()['embedding_vector']

            Answer_embedding = remote_requests.post(
                'http://37.224.68.132:27329/text_embedding/all_mpnet_base_v2',
                json = {
                  "text": r['Answer']
                }
                ).json()['embedding_vector']

            qa_pairs.append({
                'Question':r['Question'],
                'Answer':r['Answer'],
                'Question_embedding':Question_embedding,
                'Answer_embedding':Answer_embedding,
                })

        except Exception as e:
            print(e)


    ## save the results
    pd.DataFrame(qa_pairs).to_json(
        'qa_pairs_embeddings.json', 
        lines = True, 
        orient = 'records',)

    ## save the data into session
    session = uuid4()
    data = SessionData(
        filename=file.filename,
        qa_pairs=qa_pairs,
        )

    await backend.create(session, data)
    cookie.attach_to_response(response, session)

    ##

    return {
    "file_name": file.filename,
    "result": f'Uploaded and completed {len(qa_pairs)} QA pairs.',
    'running_time':time.time() - start_time,
    }



@app.post(
    "/chat_complete",
    dependencies=[Depends(cookie)],
    )
async def chat_complete(
    input:chat_input,
    session_data: SessionData = Depends(verifier),
    ):

    start_time = time.time()
    
    user_input = input.messages[-1]["content"]

    # embedding of the input
    input_embedding = remote_requests.post(
        'http://37.224.68.132:27329/text_embedding/all_mpnet_base_v2',
        json = {
        "text": user_input
        }
        ).json()['embedding_vector']


    # score the qa pairs
    similar_qas = []

    for r in session_data.qa_pairs:  

        question_score = np.dot(
            np.array(input_embedding),
            np.array(r['Question_embedding']),
            )

        # if the question matches the qa, return the answer
        if question_score >= 0.9:
            return {
            "response": r['Answer'],
            "response_source":"semantic_search",
            "response_score":question_score,
            "running_time":time.time() - start_time,
            }

        answer_score = np.dot(
            np.array(input_embedding),
            np.array(r['Answer_embedding']),
            )

        overall_score = np.max([question_score,answer_score])
        if overall_score >= 0.8:
            similar_qas.append({
                'Question':r['Question'],
                'Answer':r['Answer'],
                'question_score':question_score,
                'answer_score':answer_score,
                'overall_score':overall_score
                })

    similar_qas = sorted(similar_qas, key=lambda x: x['overall_score'],)

    # prompt engineering

    prompt_conversation = []

    for r in similar_qas[-4:]:
        prompt_conversation.append(f"[INST] {r['Question'].strip()} [/INST]")
        prompt_conversation.append(f"{r['Answer'].strip()}")

    for m in input.messages[-10:]:
        if m['role'] == 'user':
            prompt_conversation.append(f"[INST] {m['content'].strip()} [/INST]")
        else:
            prompt_conversation.append(f"{m['content'].strip()}")

    prompt_conversation = ',\n'.join(prompt_conversation)

    prompt = f"""
<<SYS>> {system_prompt} <</SYS>>

{prompt_conversation}
    """

    #print(prompt)
    response = remote_requests.post(
        'http://37.224.68.132:28074/generate',
        json = {
        "prompt": prompt,
        "stream": False,
        "max_tokens": 512,
        "top_p": 1,
        "stop": ["[INST"]
        }
        )

    response = response.json()['response'][0]
    #print(response)
    response = response.split('[INST')[0].strip()

    return {
    "response": str(response),
    "response_source":"generative_model",
    "running_time":time.time() - start_time,
    }
