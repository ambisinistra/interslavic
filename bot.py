import hashlib
import logging

from functools import wraps

from aiogram import executor, Bot, Dispatcher, types
from aiogram.types import InlineQueryResultArticle, InputTextMessageContent  # INLINE MODE!

from transformers import AutoTokenizer, M2M100ForConditionalGeneration

from api_token import API_TOKEN

# Configure logging to display on screen and save to a file
log_file = "translation_logs.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Display logs on screen
        logging.FileHandler(log_file, mode="a")  # Save logs to a file
    ]
)

bot = Bot(API_TOKEN)
dp = Dispatcher(bot)

model_name = "Salavat/nllb-200-distilled-600M-finetuned-isv_v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Logging decorator
def log_translation(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        input_text = args[0] if args else kwargs.get('input_text', 'No Input Provided')
        logging.info(f"Translation input >> {input_text}")
        
        result = await func(*args, **kwargs)
        
        logging.info(f"Translation output >> {result}")
        return result
    return wrapper

# Decorated translation function
@log_translation

async def translate_text(input_text: str) -> str:
    inputs = tokenizer(input_text, return_tensors="pt")
    output_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=5, early_stopping=True)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message) -> None:
    await message.answer("Pozdråv! Ja jesm bot, ktory prěkladjaje teksty na međuslovjansky. Ja mogu raditi v inline_modu")


@dp.message_handler()
async def text_handler(message: types.Message) -> None:
    translated_text = await translate_text(message.text)
    await message.answer(translated_text)

@dp.inline_handler()
async def inline_echo(inline_query: types.InlineQuery) -> None:
    text = inline_query.query or 'Čto ty hćeš prěkladati?'
    translated_text = await translate_text(text)
    result_id: str = hashlib.md5(text.encode()).hexdigest()
    input_content = InputTextMessageContent(translated_text, parse_mode='html')

    item = InlineQueryResultArticle(
        id=result_id,
        input_message_content=input_content,
        title='Prěkladany tekst',
        description=translated_text,
    )

    await bot.answer_inline_query(inline_query.id, results=[item], cache_time=1000)

if __name__ == '__main__':
    executor.start_polling(dispatcher=dp,
                           skip_updates=True)