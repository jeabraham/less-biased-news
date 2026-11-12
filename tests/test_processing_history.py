"""Offline regression tests using production functions with external services mocked."""

import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import re
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import news_formatter
import processing_history as history


def load_pipeline_functions(filename):
    # Avoid importing the heavyweight AI/NLP stack; execute unchanged production functions.
    path = Path(__file__).resolve().parent.parent / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    module = types.ModuleType(filename)
    module.__dict__.update(
        logger=logging.getLogger(filename), os=os, json=json, re=re, datetime=datetime, timedelta=timedelta,
        timezone=timezone,
        start_history=history.start_history, record_step=history.record_step,
        history_scope=history.history_scope, write_processing_pages=history.write_processing_pages,
        record_model_event=history.record_model_event,
        FEMALE_CONFIDENCE_THRESHOLD=0.6, FEMALE_IMAGE_STATUSES=news_formatter.FEMALE_IMAGE_STATUSES,
        get_image_size_and_url=news_formatter.get_image_size_and_url,
        get_timing_tracker=Mock(return_value=Mock()),
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module


class ProcessingHistoryTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = load_pipeline_functions("news_filter.py")
        self.nlp = Mock(return_value=types.SimpleNamespace(ents=[]))
        self.art = {"url": "https://example.com/story", "title": "A story",
                    "description": "Provider description", "body": "He led the project.\nAdvertisement"}
        self.cfg = {"prompts": {"clean_article": "Clean", "short_summary": "Summarize"}}
        self.qcfg = {"name": "Example", "fallback": "short_summary", "classification": False}
        self.stats = {"persons": 0, "classified_ok": 0, "female_names": 0, "keyword_hits": 0}
        self.pipeline.analyze_image_gender = Mock(return_value=([], (800, 600)))
        self.pipeline.clean_article = Mock(return_value="He led the project.")
        self.pipeline.short_summary = Mock(return_value="He led.")
        self.pipeline.spin_genders = Mock(return_value="She led the project.")
        self.pipeline.add_background_on_women = Mock(side_effect=lambda text, *args: text + " More background.")
        self.pipeline.clean_summary = Mock(return_value="Leader summary.")

    def process(self, images=None, summarize=True):
        self.pipeline.categorize_article_and_generate_content(
            self.art, images or [], self.cfg, self.qcfg, Mock(), self.nlp, {}, self.stats, summarize)
        return self.art["processing_history"]["steps"]

    def test_stable_id_and_legacy_cache_key(self):
        key = self.pipeline.get_article_key(self.art)
        first = history.get_article_id(self.art)
        self.art["new_today"] = False
        self.art["content"] = "Different output"
        self.assertEqual(first, history.get_article_id(self.art))
        self.assertEqual(key, "https://example.com/story||A story")
        self.art["title"] += " updated"
        self.assertNotEqual(first, history.get_article_id(self.art))

    def test_short_summary_diffs_and_decisions(self):
        steps = self.process()
        operations = [s for s in steps if "before" in s]
        self.assertEqual([s["name"] for s in operations], ["clean_article", "short_summary"])
        self.assertEqual(operations[1]["before"], "He led the project.")
        self.assertEqual(operations[1]["after"], "He led.")
        self.assertEqual(self.art["content"], "He led.")
        self.assertIn("Leadership classification skipped", [s["name"] for s in steps])

    def test_none_description_remains_safe_with_history(self):
        self.art["description"] = None
        self.qcfg.update(fallback="exclude", remove_male_pronouns=True)
        steps = self.process()
        self.assertEqual(self.art["content"], "")
        description = next(s for s in steps if s["name"] == "Use provider description")
        self.assertEqual(description["after"], "")

    def test_non_string_model_output_is_tracked_before_history_diff(self):
        self.pipeline.short_summary.return_value = None
        steps = self.process()
        self.assertEqual(self.art["content"], "")
        self.pipeline.get_timing_tracker().record_error.assert_called_with("content_not_string")
        summary = next(s for s in steps if s["name"] == "short_summary")
        self.assertIn("NoneType", summary["details"]["error"])
        self.assertEqual(summary["after"], "")

    def test_spin_pronouns_initials_and_background_each_have_diff(self):
        self.qcfg.update(fallback="spin_genders", remove_male_pronouns=True, male_initials=True)
        steps = self.process()
        self.assertEqual([s["name"] for s in steps if "before" in s], [
            "clean_article", "spin_genders", "Neutralize male pronouns",
            "Replace male names with initials", "add_background_on_women",
        ])
        self.assertTrue(self.art["content"].endswith(" More background."))

    def test_female_leader_full_body_without_summary(self):
        self.qcfg["classification"] = True
        self.cfg["female_leadership_detection"] = {"use_local_checks": False}
        self.pipeline.classify_leadership = Mock(return_value=(True, "Alice"))
        self.process(summarize=False)
        self.assertEqual(self.art["status"], "female_leader")
        self.assertEqual(self.art["content"], "He led the project.")
        self.pipeline.short_summary.assert_not_called()
        self.pipeline.clean_summary.assert_not_called()

    def test_rejection_records_response_and_stops(self):
        self.qcfg["rejection_prompt"] = "reject_test"
        self.pipeline.reject_classification = Mock(return_value=(True, "REJECT: unrelated"))
        steps = self.process(["image"])
        self.assertEqual(self.art["status"], "exclude")
        self.assertEqual(steps[-1]["details"]["reason"], "Rejection check rejected story")
        self.pipeline.clean_article.assert_not_called()
        self.pipeline.analyze_image_gender.assert_not_called()

    def test_classification_gates_and_error_are_visible(self):
        result = self.pipeline.identify_female_leadership(
            "No keywords", {}, {}, [], self.stats, Mock(), history_article=self.art)
        self.assertEqual(result, (False, None))
        history.start_history(self.art)
        self.pipeline.classify_leadership = Mock(side_effect=RuntimeError("model unavailable"))
        self.pipeline.identify_female_leadership(
            "Body", {"female_leadership_detection": {"use_local_checks": False}},
            {}, [], self.stats, Mock(), history_article=self.art)
        details = self.art["processing_history"]["steps"][-1]["details"]
        self.assertTrue(details["llm_called"])
        self.assertEqual(details["error"], "model unavailable")

    def test_image_threshold_selection_discard_and_sizing(self):
        faces = [{"gender": "Woman", "confidence": 0.59, "prominence": 0.3}]
        confident = [{"gender": "Woman", "confidence": 0.6, "prominence": 0.2}]
        self.pipeline.analyze_image_gender.side_effect = [
            (faces, (800, 600)), (confident, (800, 600)), ([], (800, 600))]
        self.qcfg["fallback_image_female"] = "spin_genders"
        steps = self.process(["low", "woman", "empty"])
        self.assertEqual(self.art["most_relevant_image"], "woman")
        self.assertEqual(self.art["status"], "spin_genders")
        selection = next(s["details"] for s in steps if s["name"] == "Image analysis and selection")
        self.assertEqual(selection["discarded_from_display"], ["low", "empty"])
        self.assertEqual(selection["images"]["low"]["raw_faces"], faces)
        self.assertEqual(selection["images"]["low"]["status"], "no_female_faces")
        presentation = steps[-1]["details"]["images"]
        self.assertEqual([i["max_width_px"] for i in presentation], [150, 400, 200])
        self.assertEqual([i["displayed"] for i in presentation], [False, True, False])

    def test_image_failure_not_reported_as_success(self):
        self.pipeline.analyze_image_gender.return_value = (None, None)
        self.process(["failed"])
        self.assertIn("failed", self.art["image_analysis"]["failed"]["error"])

    def test_worker_download_error_is_saved_in_story_history(self):
        def analyze(url, diagnostics):
            diagnostics.update(stage="image_load", error="ConnectionError: Failed to resolve 'i.cbc.ca'")
            return None, None
        self.pipeline.analyze_image_gender.side_effect = analyze
        steps = self.process(["cbc-image"])
        analysis = self.art["image_analysis"]["cbc-image"]
        self.assertEqual(analysis["diagnostics"]["stage"], "image_load")
        self.assertIn("Failed to resolve", analysis["error"])
        selection = next(s for s in steps if s["name"] == "Image analysis and selection")
        self.assertEqual(selection["details"]["images"]["cbc-image"]["error"], analysis["error"])

    def test_extraction_history_and_provider_fallback(self):
        def fetch(url, max_images, diagnostics):
            diagnostics["attempts"] = [{"method": "extractor", "error": "fetch failed"}]
            return "", []
        self.pipeline.fetch_full_text_and_images = Mock(side_effect=fetch)
        bodies, images, persons = self.pipeline.gather_bodies_images_and_persons([self.art], self.nlp)
        self.assertEqual(bodies, ["A story Provider description"])
        details = self.art["processing_history"]["steps"][0]["details"]
        self.assertTrue(details["used_provider_fallback"])
        self.assertEqual(details["extracted_text"], "")
        self.assertEqual(details["body_used"], bodies[0])
        self.assertEqual(details["image_count"], 0)
        self.assertEqual(self.art["processing_history"]["retrieved_data"]["description"], "Provider description")

    def test_html_links_escape_and_persistence(self):
        self.art["body"] = "<script>alert('unsafe')</script>"
        self.process()
        legacy = {"title": "Old", "status": "short_summary"}
        results = {"Example": [self.art, legacy]}
        with tempfile.TemporaryDirectory() as directory:
            history.write_processing_pages(results, directory)
            path = Path(directory) / self.art["processing_page"]
            content = path.read_text(encoding="utf-8")
            self.assertIn("&lt;script&gt;", content)
            self.assertNotIn("<script>", content)
            self.assertIn("Text diff", content)
            self.assertIn("shortened", content)
            self.assertIn("He led.", content)
            before = path.stat().st_mtime_ns
            history.write_processing_pages(results, directory)
            self.assertEqual(before, path.stat().st_mtime_ns)
            self.assertIn(self.art["processing_page"], news_formatter.generate_html(results))
            self.assertNotIn("story-", news_formatter.render_article_to_html(legacy))
            self.assertNotIn("processing_page", legacy)
            path.unlink()
            history.write_processing_pages(results, directory)
            self.assertTrue(path.exists())

    def test_cache_reuse_skips_all_article_processing_and_retains_history(self):
        self.process()
        cached = deepcopy(self.art)
        self.pipeline.NewsApiClient = Mock()
        self.pipeline.spacy = Mock(load=Mock(return_value=self.nlp))
        self.pipeline.Genderize = Mock()
        self.pipeline.AIUtils = Mock()
        self.pipeline.load_cache = Mock(return_value={"articles": {
            self.pipeline.get_article_key(cached): {"article_data": cached}}})
        self.pipeline.fetch_articles = Mock(return_value=[{
            "title": self.art["title"], "url": self.art["url"], "description": "New provider snippet"}])
        self.pipeline.save_cache = Mock()
        self.pipeline.fetch_full_text_and_images = Mock()
        self.pipeline.guess_genders = Mock()
        self.pipeline.categorize_article_and_generate_content = Mock()
        cfg = {"newsapi": {"api_key": "not-real"}, "queries": [self.qcfg]}
        result = self.pipeline.fetch_and_filter(cfg, new_today=True)["Example"][0]
        self.pipeline.fetch_full_text_and_images.assert_not_called()
        self.pipeline.guess_genders.assert_not_called()
        self.pipeline.categorize_article_and_generate_content.assert_not_called()
        self.assertEqual(result["processing_history"], cached["processing_history"])
        self.assertEqual(result["body"], cached["body"])
        self.assertFalse(result["new_today"])

    def test_publication_age_and_unknown_dates(self):
        cutoff = datetime(2026, 1, 8, 12, tzinfo=timezone.utc)
        recent = self.pipeline.is_recent_article
        self.assertTrue(recent({"publishedAt": "2026-01-08T12:00:00Z"}, cutoff))
        self.assertFalse(recent({"published_at": "2026-01-08T13:59:59+02:00"}, cutoff))
        self.assertTrue(recent({"published_at": "2026-01-08T12:00:00"}, cutoff))
        with self.assertLogs("news_filter.py", level="WARNING") as logs:
            self.assertTrue(recent({"publishedAt": "invalid"}, cutoff))
            self.assertTrue(recent({}, cutoff))
        self.assertEqual(len(logs.output), 2)

    def test_unseen_old_articles_are_processed_but_not_new(self):
        self.pipeline.NewsApiClient = Mock()
        self.pipeline.spacy = Mock(load=Mock(return_value=self.nlp))
        self.pipeline.Genderize = Mock()
        self.pipeline.AIUtils = Mock()
        now = datetime.now(timezone.utc)
        articles = [
            {"title": "Old", "url": "old", "published_at": (now - timedelta(days=8)).isoformat()},
            {"title": "Recent", "url": "recent", "publishedAt": (now - timedelta(days=6)).isoformat()},
            {"title": "Cached", "url": "cached", "publishedAt": now.isoformat()},
        ]
        self.pipeline.fetch_articles = Mock(return_value=deepcopy(articles))
        self.pipeline.gather_bodies_images_and_persons = Mock(
            side_effect=lambda pending, *args, **kwargs: (["body"] * len(pending), [[] for _ in pending], set()))
        self.pipeline.categorize_article_and_generate_content = Mock(
            side_effect=lambda art, *args: art.update(status="short_summary", content=art["title"]))
        cfg = {"newsapi": {"api_key": "not-real"}, "queries": [self.qcfg]}
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "cache.json")
            self.pipeline.get_cache_file_path = Mock(return_value=path)
            self.pipeline.save_cache(path, {"articles": [articles[2]]})
            result = self.pipeline.fetch_and_filter(cfg, new_today=True)
            self.assertEqual([a["new_today"] for a in result["Example"]], [False, True, False])
            self.assertEqual(self.pipeline.categorize_article_and_generate_content.call_count, 2)
            self.assertEqual([a["title"] for _, a in news_formatter.get_new_today_articles(result)], ["Recent"])
            self.assertIn("Old", news_formatter.generate_html(result))
            cached = self.pipeline.load_cache(path)["articles"]
            self.assertNotIn("_reprocessed", cached["old||Old"]["article_data"])
            self.pipeline.fetch_articles.return_value = deepcopy(articles)
            self.pipeline.fetch_and_filter(cfg, new_today=True)
            self.assertEqual(self.pipeline.categorize_article_and_generate_content.call_count, 2)
            self.pipeline.get_cache_file_path.return_value = str(Path(directory) / "other.json")
            self.pipeline.fetch_articles.return_value = deepcopy(articles[:1])
            result = self.pipeline.fetch_and_filter({**cfg, "not_new_if_older_than": 9}, new_today=True)
            self.assertTrue(result["Example"][0]["new_today"])

    def test_cache_retention_default_config_and_override(self):
        self.pipeline.NewsApiClient = Mock()
        self.pipeline.spacy = Mock(load=Mock(return_value=self.nlp))
        self.pipeline.Genderize = Mock()
        self.pipeline.AIUtils = Mock()
        self.pipeline.fetch_articles = Mock(return_value=[])
        self.pipeline.gather_bodies_images_and_persons = Mock(return_value=([], [], set()))
        cfg = {"newsapi": {"api_key": "not-real"}, "queries": [self.qcfg]}
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "cache.json")
            self.pipeline.get_cache_file_path = Mock(return_value=path)
            def seed():
                entries = {
                    str(days): {"first_seen": (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(),
                                "article_data": {"title": str(days)}}
                    for days in (10, 59, 61)
                }
                Path(path).write_text(json.dumps({"articles": entries}))
            seed()
            self.pipeline.fetch_and_filter(cfg)
            self.assertEqual(set(self.pipeline.load_cache(path)["articles"]), {"10", "59"})
            seed()
            cfg["cache_expire_days"] = 20
            self.pipeline.fetch_and_filter(cfg)
            self.assertEqual(set(self.pipeline.load_cache(path)["articles"]), {"10"})
            seed()
            self.pipeline.fetch_and_filter(cfg, expire_days=65)
            self.assertEqual(set(self.pipeline.load_cache(path)["articles"]), {"10", "59", "61"})
        with self.assertRaises(ValueError):
            self.pipeline.fetch_and_filter({**cfg, "not_new_if_older_than": -1})

    def test_model_truncation_and_refusal_are_recorded(self):
        queries = load_pipeline_functions("ai_queries.py")
        ai = Mock()
        ai._call_ollama_api.return_value = "I cannot assist"
        history.start_history(self.art)
        with history.history_scope(self.art):
            result = queries.ollama_call("Prompt", "Long input", 20,
                                        {"ollama": {"max_input_chars": 4}}, ai, task="spin_genders")
        self.assertEqual(result, "")
        steps = self.art["processing_history"]["steps"]
        self.assertEqual(steps[0]["details"]["discarded_characters"], 6)
        self.assertTrue(steps[1]["details"]["refused"])
        self.assertEqual(ai._call_ollama_api.call_args.args[0], "Prompt\n\nLong")

    def test_multiple_query_histories_share_one_story_page(self):
        self.process()
        other = deepcopy(self.art)
        history.record_step(other, "Different query", {"query": "Second"})
        with tempfile.TemporaryDirectory() as directory:
            history.write_processing_pages({"First": [self.art], "Second": [other]}, directory)
            self.assertEqual(self.art["processing_page"], other["processing_page"])
            self.assertEqual(len(list(Path(directory).glob("*.html"))), 1)
            self.assertIn("Different query", (Path(directory) / other["processing_page"]).read_text())

    def test_cached_subset_does_not_replace_processing_page(self):
        self.process()
        other = deepcopy(self.art)
        history.record_step(other, "Second query", {})
        with tempfile.TemporaryDirectory() as directory:
            history.write_processing_pages({"First": [self.art], "Second": [other]}, directory)
            path = Path(directory) / self.art["processing_page"]
            content, mtime = path.read_text(), path.stat().st_mtime_ns
            self.art["new_today"] = False
            history.write_processing_pages({"First": [self.art]}, directory)
            self.assertEqual(path.read_text(), content)
            self.assertEqual(path.stat().st_mtime_ns, mtime)

    def test_old_reprocessed_article_replaces_processing_page(self):
        self.process()
        with tempfile.TemporaryDirectory() as directory:
            history.write_processing_pages({"Example": [self.art]}, directory)
            path = Path(directory) / self.art["processing_page"]
            history.record_step(self.art, "Reprocessed old article", {})
            self.art.update(new_today=False, _reprocessed=True)
            history.write_processing_pages({"Example": [self.art]}, directory)
            self.assertIn("Reprocessed old article", path.read_text())

    def test_filter_file_output_has_resolvable_status_link(self):
        self.process()
        self.pipeline.setup_logging = Mock()
        self.pipeline.load_config = Mock(return_value={"queries": [self.qcfg]})
        self.pipeline.fetch_and_filter = Mock(return_value={"Example": [self.art]})
        self.pipeline.generate_html = news_formatter.generate_html
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-output" / "daily.html"
            self.pipeline.main(output=str(output), fmt="html")
            page = self.art["processing_page"]
            self.assertIn(f"href='{page}'", output.read_text())
            self.assertTrue((output.parent / page).exists())

    def test_standalone_formatter_creates_pages_from_cache(self):
        self.process()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "Example_cache.json"
            cache.write_text(json.dumps({"articles": {"key": {"article_data": self.art}}}))
            output = Path(directory) / "summary.html"
            with patch.object(news_formatter, "find_query_files", return_value={"Example": str(cache)}), \
                    patch.object(sys, "argv", ["news_formatter.py", "--format", "html", "--output", str(output)]):
                news_formatter.main()
            page = f"story-{history.get_article_id(self.art)}.html"
            self.assertIn(page, output.read_text())
            self.assertTrue((output.parent / page).exists())

    def test_extractor_attempts_and_limit_omissions(self):
        fetcher = load_pipeline_functions("article_fetcher.py")
        fetcher.Article = Mock(side_effect=RuntimeError("newspaper unavailable"))
        fetcher.requests = Mock()
        fetcher.Document = Mock()
        soup = Mock()
        soup.get_text.return_value = "Retrieved body"
        soup.find_all.return_value = [{"src": "one"}, {"src": "two"}, {"src": "three"}]
        soup.find.return_value = None
        fetcher.BeautifulSoup = Mock(return_value=soup)
        diagnostics = {}
        text, images = fetcher.fetch_full_text_and_images("https://example.com/story",
                                                         max_images=2, diagnostics=diagnostics)
        self.assertEqual(text, "Retrieved body")
        self.assertEqual(images, ["one", "two"])
        self.assertEqual(diagnostics["selected_method"], "fetch_with_readability")
        self.assertEqual(diagnostics["attempts"][0]["error"], "newspaper unavailable")
        self.assertEqual(diagnostics["attempts"][1]["candidate_images"], ["one", "two", "three"])
        self.assertEqual(diagnostics["attempts"][1]["omitted_by_max_images"], ["three"])

    def test_cache_serialization_retains_full_history(self):
        self.process()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "Example_cache.json")
            self.pipeline.save_cache(path, {"articles": [self.art]})
            loaded = self.pipeline.load_cache(path)["articles"][self.pipeline.get_article_key(self.art)]
            self.assertEqual(loaded["article_data"]["processing_history"], self.art["processing_history"])

    def test_local_summary_truncated_fallback_is_explained(self):
        summarization = load_pipeline_functions("summarization.py")
        summarization.pipeline = Mock(side_effect=RuntimeError("model missing"))
        history.start_history(self.art)
        with history.history_scope(self.art):
            result = summarization.summarize_text_using_local_model("Long text", max_length=1)
        self.assertEqual(result, "Long...")
        details = self.art["processing_history"]["steps"][0]["details"]
        self.assertEqual(details["character_limit"], 4)
        self.assertEqual(details["error"], "model missing")


if __name__ == "__main__":
    unittest.main()
