import pickle
import pandas as pd

def extrair_pares_history(caminho_history_pkl):
    """
    Carrega o arquivo history.pkl e alinha prompts e ações por passo e por ambiente.
    """
    with open(caminho_history_pkl, "rb") as f:
        history = pickle.load(f)
    
    registros = []

    print([(key, len(history[key])) for key in history])
    
    def print_structure(obj, path="history", indent=0):
        prefix = " " * indent

        if isinstance(obj, dict):
            print(f"{prefix}{path} -> dict, len={len(obj)}")

            for key, value in obj.items():
                child_path = f'{path}[{key!r}]'
                print_structure(value, child_path, indent + 1)

        elif isinstance(obj, (list, tuple)):
            print(f"{prefix}{path} -> {type(obj).__name__}, len={len(obj)}")

            # Print the structure of the first element only.
            # Otherwise large lists such as history["prompts"] would generate
            # thousands of lines.
            if len(obj) > 0:
                print_structure(obj[0], f"{path}[0]", indent + 1)

        else:
            print(f"{prefix}{path} -> {type(obj).__name__}")

    print_structure(history)

    # total_passos = len(history['prompts'])
    
    # for passo_idx in range(total_passos):
    #     prompts_passo = history['prompts'][passo_idx]
    #     acoes_passo = history['actions'][passo_idx]
    #     acoes_possiveis_passo = history['possible_actions'][passo_idx] if 'possible_actions' in history else [None] * len(prompts_passo)
        
    #     # Intercala cada ambiente simulado em paralelo naquele passo t
    #     for env_idx, (prompt, acao) in enumerate(zip(prompts_passo, acoes_passo)):
    #         registros.append({
    #             "passo": passo_idx,
    #             "env_id": env_idx,
    #             "prompt": prompt,
    #             "acao_tomada": acao,
    #             "acoes_possiveis": acoes_possiveis_passo[env_idx]
    #         })
            
    # df = pd.DataFrame(registros)
    # return df

import pickle
import pandas as pd

def extrair_pares_replay_buffer(caminho_buffer_pkl):
    """
    Carrega o NStepReplayBuffer lendo a partir do atributo rb.memory.
    """
    with open(caminho_buffer_pkl, "rb") as f:
        rb = pickle.load(f)
    
    registros = []
    
    # Itera sobre os elementos salvos em memory
    for item in rb.memory:
        if isinstance(item, (tuple, list)):
            # Mapeia a ordem exata do método buffer.add(...) do seu script:
            # 0: prompt, 1: acao, 2: recompensa, 3: proximo_prompt, 4: done, 5: acoes_possiveis
            registros.append({
                "prompt": item[0],
                "acao_tomada": item[1],
                "recompensa": item[2],
                "proximo_prompt": item[3],
                "done": item[4],
                "acoes_possiveis": item[5] if len(item) > 5 else None
            })
        elif isinstance(item, dict):
            registros.append(item)
            
    return pd.DataFrame(registros)

# Exemplo de uso:
df_historico = extrair_pares_history("/media/gustavo/hd1/outputs/magellan/463032/history.pkl")
# for i in range(df_historico["env_id"].max()):
#     print(df_historico[df_historico["env_id"]==i])
#     df_historico[df_historico["env_id"]==i].to_csv('episode.csv', index=False)
#     exit()

# df_buffer = extrair_pares_replay_buffer("/media/gustavo/hd1/outputs/magellan/463032/replay_buffer.pkl")
# print(df_buffer.head())
# df_buffer.head().to_csv('episode.csv', index=False)