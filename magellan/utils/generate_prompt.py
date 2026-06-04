'''

'''

def generate_prompt(o, g, *args, **kwargs):
    prompt = f'Goal: {g}\n'
    prompt += o
    prompt += '\nAction: '
    return prompt