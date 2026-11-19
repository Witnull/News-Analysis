from datetime import datetime
import logging

from src.modules.llm.llmPool import LLMPool
from src.modules.collector.cleaner import ArticleCleaner
from src.modules.collector.filter import ArticleFilter

from src.modules.store.store import ArticleStore
import traceback
import os
import asyncio

from src.utils.logger import setup_logger

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.environ.get("LOG_DIR", None)
if LOG_DIR is None:
    LOG_DIR = f"./logs/Man_{timestamp}"
    os.environ["LOG_DIR"] = LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the log directory exists
logger = setup_logger(
    f"{LOG_DIR}/feedManager.log", log_level=logging.DEBUG
)  # Adjust log level as needed


VERBOSE = False  # Set to True for more detailed logs
DEBUG = False  # Set to True to save debug outputs for inspection
DEBUG_DIR = "./debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

TRANSLATE = False # translate or not
SAVE = False # save to db or not


class FeedManager:
    def __init__(
        self,
        db_url: str = "sqlite:///articles_test2.db",
        timeframe_days: int = 7,
        translate: bool = TRANSLATE,
        save: bool = SAVE,
    ):
        self.art_store = ArticleStore(db_url)

        self.filter = LLMPool(ArticleFilter, n_workers=2, verbose=False)
        self.cleaner = ArticleCleaner(verbose=True) #LLMPool(ArticleCleaner, n_workers=2, verbose=VERBOSE)

        self.timeframe_days = timeframe_days
        self.articles = self.art_store.load_all(to_review=True)[:1000]
        # self.art_store.load_all_by_pubdate(date=self.timeframe_days)
        self.translate = translate
        self.save = save
        logger.info(f"[YELLOW] Logs saved to: {LOG_DIR}")
        if not self.save:
            logger.info(
                "[YELLOW] Warning: Articles will not be saved to the database. This is intended for testing purposes only."
            )
        if DEBUG:
            logger.info(
                f"[YELLOW] Debug mode is ON. Articles will be saved to {DEBUG_DIR} for inspection."
            )

    async def run(self):

        logger.info("== FILTER ==================================")
        try:
            self.filter.load()
            tasks_1 = [
                self.filter.call("process_single", article) for article in self.articles #if article.processed_stage == "none" or not article.is_reviewed
            ]
            await asyncio.gather(*tasks_1)
            
        except Exception as e:
            logger.info(f"Error during filtering: {e}")
            traceback.print_exc()
        finally:
            self.filter.unload()
        # try:
        #     with self.filter:
        #         self.filter.process_batch(self.articles)
        # except Exception as e:
        #     logger.info(f"Error during filtering: {e}")
        #     traceback.print_exc()

        keep_art = [art for art in self.articles if art.no_save_pool.get("keep", False) ]#or art.is_reviewed]
        for art in keep_art:
            art.processed_stage = "filtered"

        for art in self.articles:
            if not (art.no_save_pool.get("keep", False) ): #and art.is_reviewed):
                logger.info(
                    f"[YELLOW] Discarded: {art.title} (reason: {art.no_save_pool.get('filter_reason', 'unknown')})"
                )
                if self.save:
                    if self.art_store.move(art.url, reason=art.no_save_pool.get('filter_reason', 'unknown')):
                        logger.info(f"[YELLOW][SAVE ON] Moved to discarded: {art.title}")

        logger.info("== CLEAN ==================================")
        # region clean
        # try:
        #     self.cleaner.load()
        #     tasks_2 = [
        #         self.cleaner.call("process_single", article) for article in keep_art
        #     ]
        #     await asyncio.gather(*tasks_2)
        #     self.cleaner.unload()
        # except Exception as e:
        #     logger.info(f"Error during cleaning: {e}")
        #     traceback.logger.info_exc()
        if len([art for art in keep_art if art.no_save_pool.get('need_cleaning', False)]) != 0:   
            try:
                with self.cleaner:
                    self.cleaner.process_batch([art for art in keep_art if art.no_save_pool.get("need_cleaning", False) ])# and art.processed_stage in ["filtered"]])
            except Exception as e:
                logger.info(f"Error during cleaning: {e}")
                traceback.print_exc()
        else:
            logger.info("[YELLOW] No articles need cleaning.")

        for art in keep_art:
            art.is_reviewed = True  # mark as reviewed after cleaning

        print(f"[YELLOW] Articles after cleaning: {len(keep_art)}")
        if self.save:
            logger.info(f"[YELLOW][SAVE ON] Saving {len(keep_art)} articles after cleaning.")
            self.art_store.save_many(keep_art)


        # logger.info("== TRANSLATE ==================================")
        # # region translate
        # if self.translate:
        #     with self.translator:
        #         try:
        #             art_2 = self.translator.process_batch(keep_art)
        #         except Exception as e:
        #             logger.info(f"Error during translation: {e}")

        # if DEBUG:
        #     dx = []
        #     for art in keep_art:
        #         if art.no_save_pool.get("translated", None) is not None:
        #             dx.append(art.to_dict())
        #     json.dump(
        #         dx,
        #         open(
        #             os.path.join(DEBUG_DIR, "articles_translated.json"),
        #             "w",
        #             encoding="utf-8",
        #         ),
        #         indent=2,
        #         ensure_ascii=False,
        #     )

        # logger.info("== SUMMARIZE ==================================")
        # # region summarize
        # with self.summarizer:
        #     try:
        #         self.summarizer.process_batch([x for x in keep_art if x.summarized_text is None or x.summarized_text.strip() == "" or x.processed_stage in ["filtered", "cleaned", "translated"]])
        #     except Exception as e:
        #         logger.info(f"Error during summarization: {e}")
        #         traceback.print_exc()
        # if self.save:
        #     self.art_store.save_many(keep_art)

        # if DEBUG:
        #     dx = []
        #     for art in self.articles:
        #         dx.append(art.to_dict())
        #     json.dump(dx, open(os.path.join(DEBUG_DIR, "articles_summarized.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

        # logger.info("== CONCEPT EXT ==================================")
        # # region concept extract
        # with ArticleConceptExtractor() as concept_extractor:
        #     try:
        #         concept_extractor.process_batch([x for x in keep_art if x.processed_stage in ["summarized"]])
        #     except Exception as e:
        #         logger.info(f"Error during concept extraction: {e}")
        #         traceback.print_exc()

        # if self.save:
        #     self.art_store.save_many(keep_art)

        # logger.info("== WORTHY EVAL ==================================")
        # # region worthy eval
        # with self.worthy_eval:
        #     try:
        #         self.worthy_eval.process_batch([x for x in keep_art if x.yield_score is None or x.processed_stage in ["summarized", "concept_extracted"]])
        #     except Exception as e:
        #         logger.info(f"Error during curation: {e}")
        #         traceback.print_exc()
        # # region filter2
        # # strategy - keep over 4 - discard 3 and below
        # if self.save:
        #     self.art_store.save_many(keep_art)
            
        # prev_keep_art = keep_art
        # keep_art = [
        #     art for art in prev_keep_art
        #     if art.yield_score >= 4
        # ]
        # for art in prev_keep_art:
        #     if not art.no_save_pool.get("keep", False) or art.yield_score < 4:
        #         logger.info(f"Discarded: {art.title} (reason: Low yield score {art.yield_score})")
        #         if self.save:
        #             self.art_store.move(art.url, reason="Low yield score")

        # logger.info("== IMPACT EVAL ==================================")
        # # region impact eval
        # with self.impact_eval:
        #     try:
        #         self.impact_eval.process_batch([x for x in keep_art if x.impact_score is None or x.processed_stage in ["summarized", "worthy_evaluated", "concept_extracted"]])
        #     except Exception as e:
        #         logger.info(f"Error during impact evaluation: {e}")

        # if self.save:
        #     self.art_store.save_many(keep_art)
        # if DEBUG:
        #     dx = []
        #     for art in keep_art:
        #         dx.append(art.to_dict())
        #     json.dump(
        #         dx,
        #         open(
        #             os.path.join(DEBUG_DIR, "articles_curated.json"),
        #             "w",
        #             encoding="utf-8",
        #         ),
        #         indent=2,
        #         ensure_ascii=False,
        #     )

        # # region enrich
        # logger.info("== ENRICH ==================================")
