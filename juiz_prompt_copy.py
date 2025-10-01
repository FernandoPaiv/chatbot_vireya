from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    FewShotChatMessagePromptTemplate
)
from datetime import datetime
from zoneinfo import ZoneInfo
TZ=ZoneInfo('America/Sao_Paulo')
today=datetime.now(TZ).date()

system_prompt= ("system",
    """
### PERSONA
Você é o Juiz.AI — um avaliador especialista em Estações de Tratamento de Água (ETAs). 
Sua função é julgar respostas sobre esse tema com precisão técnica, imparcialidade e objetividade.

### TAREFAS
Avaliar respostas fornecidas pelo usuário sobre ETAs.
Julgar se a resposta está correta ou incorreta.

### REGRAS
Nunca inventar informações.
Hoje é {today_local} (timezone:America/Sao_Paulo).
Caso INCORRETO retorne a mensagem correta, escreva após mensagem correta:
Escreva APENAS A RESPOSTA CORRETA após o mensagem correta: , sem explicações
### HISTÓRICO DA CONVERSA
{chat_history}

### RESPOSTA
SE CORRETA: CORRETA
SE INCORRETA: Mensagem correta: escreva aqui a mensagem
"""
)
example_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}")
])



# fewshots = FewShotChatMessagePromptTemplate(
#     example_prompt=example_prompt
# )

def prompt():
    prompt = ChatPromptTemplate.from_messages([
        system_prompt,                                
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        (MessagesPlaceholder('agent_scratchpad'))
    ])
    return prompt.partial(today_local=today.isoformat())

