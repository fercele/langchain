import unittest

from chat_persistence.chat import Chat

class TestChat(unittest.TestCase):

    def test_cria_chat(self):
        user_id = "Laura"

        novo_chat = Chat(user_id)

        self.assertIsNotNone(novo_chat.chat_key)
        self.assertEqual(novo_chat.user_id, user_id)
        self.assertEqual(novo_chat.messages, [])

        chat_existente = Chat(user_id, novo_chat.chat_key)

        self.assertEqual(chat_existente.chat_key, novo_chat.chat_key)
        self.assertEqual(chat_existente.user_id, user_id)
        self.assertEqual(chat_existente.messages, [])

        with self.assertRaises(Exception):
            Chat("nao_sou_a_laura", novo_chat.chat_key)
        
        with self.assertRaises(Exception):
            Chat(user_id, "nao_existo")

        novo_chat.add_message(("Bom dia!", "Bom dia, em que posso ajudar?"))

        self.assertEqual(len(novo_chat.messages), 1)

        chat_existente2 = Chat(user_id, novo_chat.chat_key)
        self.assertListEqual(chat_existente2.messages, novo_chat.messages)

        primeira_mensagem = chat_existente2.messages[0]
        self.assertEqual(primeira_mensagem["pergunta"], "Bom dia!")
        self.assertEqual(primeira_mensagem["resposta"], "Bom dia, em que posso ajudar?")

        novo_chat.add_message(("Qual o seu nome?", "Sou a VictorIA, sua assistente virtual"))
        self.assertEqual(len(novo_chat.messages), 2)

        chat_existente3 = Chat(user_id, novo_chat.chat_key)
        self.assertListEqual(chat_existente3.messages, novo_chat.messages)

        segunda_mensagem = chat_existente3.messages[1]
        self.assertEqual(segunda_mensagem["pergunta"], "Qual o seu nome?")
        self.assertEqual(segunda_mensagem["resposta"], "Sou a VictorIA, sua assistente virtual")

        novo_chat.set_message_feedback(0, True, None)
        novo_chat.set_message_feedback(1, False, "Não respondeu, apesar de constar nos documentos")

        primeira_mensagem = novo_chat.messages[0]
        self.assertEqual(primeira_mensagem["feedback"]["type"], True)
        self.assertEqual(primeira_mensagem["feedback"]["message"], None)
        segunda_mensagem = novo_chat.messages[1]
        self.assertEqual(segunda_mensagem["feedback"]["type"], False)
        self.assertEqual(segunda_mensagem["feedback"]["message"], "Não respondeu, apesar de constar nos documentos")
                
        chat_existente4 = Chat(user_id, novo_chat.chat_key)
        self.assertListEqual(chat_existente4.messages, novo_chat.messages)
        primeira_mensagem = chat_existente4.messages[0]
        self.assertEqual(primeira_mensagem["feedback"]["type"], True)
        self.assertEqual(primeira_mensagem["feedback"]["message"], None)
        
        segunda_mensagem = chat_existente4.messages[1]
        self.assertEqual(segunda_mensagem["feedback"]["type"], False)
        self.assertEqual(segunda_mensagem["feedback"]["message"], "Não respondeu, apesar de constar nos documentos")


        messages_raw = novo_chat.get_messages(raw=True)
        self.assertEqual(len(messages_raw), 2)
        self.assertEqual(messages_raw[0]["pergunta"], "Bom dia!")
        self.assertEqual(messages_raw[0]["resposta"], "Bom dia, em que posso ajudar?")
        self.assertEqual(messages_raw[0]["feedback"]["type"], True)
        self.assertEqual(messages_raw[0]["feedback"]["message"], None)

        messages_notraw = novo_chat.get_messages(raw=False)
        self.assertEqual(len(messages_notraw), 2)
        
        self.assertEqual(messages_notraw[0][0], "Bom dia!")
        self.assertEqual(messages_notraw[0][1], "Bom dia, em que posso ajudar?")

        self.assertEqual(messages_notraw[1][0], "Qual o seu nome?")
        self.assertEqual(messages_notraw[1][1], "Sou a VictorIA, sua assistente virtual")
        

if __name__ == '__main__':
    unittest.main()
