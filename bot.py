import hashlib
import logging

from aiogram import executor, Bot, Dispatcher, types
from aiogram.types import InlineQueryResultArticle, InputTextMessageContent  # INLINE MODE!

from transformers import AutoTokenizer, M2M100ForConditionalGeneration

from api_token import API_TOKEN

# Настройка логирования
logging.basicConfig(level=logging.INFO)

bot = Bot(API_TOKEN)
dp = Dispatcher(bot)

model_name = "Salavat/nllb-200-distilled-600M-finetuned-isv_v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

async def translate_text(input_text: str) -> str:
    inputs = tokenizer(input_text, return_tensors="pt")
    output_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=5, early_stopping=True)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message) -> None:
    await message.answer("Hi! I'm a bot translating texts into interslavic. I can work in inline_mode")


@dp.message_handler()
async def text_handler(message: types.Message) -> None:
    translated_text = await translate_text(message.text)
    await message.answer(translated_text)

@dp.inline_handler()
async def inline_echo(inline_query: types.InlineQuery) -> None:
    text = inline_query.query or 'What do you want to translate?'
    translated_text = await translate_text(text)
    result_id: str = hashlib.md5(text.encode()).hexdigest()
    input_content = InputTextMessageContent(translated_text, parse_mode='html')

    item = InlineQueryResultArticle(
        id=result_id,
        input_message_content=input_content,
        title='Translated Text',
        description=translated_text,
    )

    await bot.answer_inline_query(inline_query.id, results=[item], cache_time=1000)

if __name__ == '__main__':
    executor.start_polling(dispatcher=dp,
                           skip_updates=True)