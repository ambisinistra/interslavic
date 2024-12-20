from api_token import API_TOKEN

from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineQuery, InputTextMessageContent, InlineQueryResultArticle
from aiogram.utils import executor
import logging
import uuid

# Настройка логирования
logging.basicConfig(level=logging.INFO)

from transformers import AutoTokenizer, M2M100ForConditionalGeneration

model_name = "Salavat/nllb-200-distilled-600M-finetuned-isv_v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# Обработчик inline-запросов
@dp.inline_handler()
async def inline_echo(inline_query: InlineQuery):
    query_text = inline_query.query or "Напишите что-нибудь!"

    # Создаем результат inline-запроса
    result = InlineQueryResultArticle(
        id=str(uuid.uuid4()),
        title="Эхо",
        input_message_content=InputTextMessageContent(query_text)
    )

    # Отправляем результат
    await inline_query.answer([result], cache_time=1)

# Обработчик обычных сообщений
@dp.message_handler()
async def cmd_start(message: types.Message):
    inputs = tokenizer(message.text, return_tensors="pt")
    
    # Generate text
    output_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=5, early_stopping=True)

    # Decode the output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    await message.answer(generated_text)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
