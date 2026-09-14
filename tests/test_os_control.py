import unittest

from server.llm.os_control import (
    OS_CONTROL_TOOL,
    ack_block,
    describe_action,
    sanitize_action,
    sanitize_os_snapshot,
    should_direct,
    wants_os_action,
)


class OsControlTests(unittest.TestCase):
    def test_tool_shape(self):
        fn = OS_CONTROL_TOOL["function"]
        self.assertEqual(fn["name"], "control_os")
        props = fn["parameters"]["properties"]["actions"]["items"]["properties"]
        self.assertIn("op", props)
        self.assertIn("open_app", props["op"]["enum"])
        self.assertIn("browser_navigate", props["op"]["enum"])
        self.assertIn("browser_new_tab", props["op"]["enum"])
        self.assertIn("browser_reload", props["op"]["enum"])
        self.assertIn("browser_close_tab", props["op"]["enum"])

    def test_should_direct_gates_trivia(self):
        self.assertFalse(should_direct("hi"))
        self.assertFalse(should_direct("नमस्ते"))
        self.assertFalse(should_direct(""))
        self.assertTrue(should_direct("browser me python docs kholo"))

    def test_snapshot_is_bounded(self):
        snap = sanitize_os_snapshot({
            "app": "browser",
            "open": ["browser", "notes", "evil", "browser"],
            "minimized": ["code"],
            "browserUrl": "https://x.com/" + "y" * 900,
            "whiteboardSteps": 5,
            "code": {"huge": "x" * 5000},
        })
        self.assertEqual(snap["app"], "browser")
        self.assertNotIn("evil", snap["open"])
        self.assertEqual(snap["minimized"], ["code"])
        self.assertLessEqual(len(snap["browserUrl"]), 500)
        self.assertEqual(snap["whiteboardSteps"], 5)
        self.assertNotIn("code", snap)  # code context never leaves the client
        self.assertEqual(sanitize_os_snapshot("junk"), {})

    def test_action_allowlist(self):
        self.assertEqual(
            sanitize_action({"op": "open_app", "app": "browser", "evil": 1}),
            {"op": "open_app", "app": "browser"},
        )
        self.assertEqual(
            sanitize_action({"op": "tile_app", "app": "code", "zone": "tl"}),
            {"op": "tile_app", "app": "code", "zone": "tl"},
        )
        self.assertIsNone(sanitize_action({"op": "tile_app", "app": "code", "zone": "center"}))
        self.assertIsNone(sanitize_action({"op": "open_app", "app": "terminal"}))
        self.assertIsNone(sanitize_action({"op": "rm_rf", "app": "browser"}))
        self.assertIsNone(sanitize_action("junk"))

    def test_browser_noarg_ops(self):
        self.assertEqual(sanitize_action({"op": "browser_back"}), {"op": "browser_back"})
        self.assertEqual(sanitize_action({"op": "browser_forward"}), {"op": "browser_forward"})
        self.assertEqual(sanitize_action({"op": "browser_new_tab"}), {"op": "browser_new_tab"})
        self.assertEqual(sanitize_action({"op": "browser_reload"}), {"op": "browser_reload"})
        self.assertEqual(sanitize_action({"op": "browser_close_tab"}), {"op": "browser_close_tab"})
        # extra keys are stripped, never forwarded
        self.assertEqual(
            sanitize_action({"op": "browser_reload", "target": "x", "app": "browser"}),
            {"op": "browser_reload"},
        )

    def test_browser_navigate_rejects_schemes(self):
        self.assertIsNone(sanitize_action({"op": "browser_navigate", "target": "javascript:alert(1)"}))
        self.assertIsNone(sanitize_action({"op": "browser_navigate", "target": "  DATA:text/html,x"}))
        self.assertIsNone(sanitize_action({"op": "browser_navigate", "target": "   "}))
        good = sanitize_action({"op": "browser_navigate", "target": "python docs"})
        self.assertEqual(good, {"op": "browser_navigate", "target": "python docs"})

    def test_note_add_bounds(self):
        self.assertIsNone(sanitize_action({"op": "note_add"}))
        note = sanitize_action({"op": "note_add", "title": "T" * 200, "body": "B" * 5000})
        self.assertEqual(len(note["title"]), 80)
        self.assertEqual(len(note["body"]), 2000)

    def test_snapshot_bounds_browser_tabs(self):
        snap = sanitize_os_snapshot({"app": "browser", "browserUrl": "https://x.com/", "browserTabs": 3})
        self.assertEqual(snap["browserTabs"], 3)
        snap = sanitize_os_snapshot({"browserTabs": 999})
        self.assertEqual(snap["browserTabs"], 32)
        snap = sanitize_os_snapshot({"browserTabs": 0})
        self.assertNotIn("browserTabs", snap)
        snap = sanitize_os_snapshot({"browserTabs": "junk"})
        self.assertNotIn("browserTabs", snap)

    def test_help_app_allowlisted(self):
        self.assertEqual(
            sanitize_action({"op": "open_app", "app": "help"}),
            {"op": "open_app", "app": "help"},
        )
        self.assertEqual(describe_action({"op": "open_app", "app": "help"}), "opened the Help Center")
        snap = sanitize_os_snapshot({"app": "help", "open": ["browser", "help"]})
        self.assertEqual(snap["app"], "help")
        self.assertIn("help", snap["open"])

    def test_action_gate(self):
        self.assertTrue(wants_os_action("browser kholo"))
        self.assertTrue(wants_os_action("notes me ye likh do"))
        self.assertTrue(wants_os_action("python docs dikhao"))
        self.assertTrue(wants_os_action("page reload karo"))
        self.assertTrue(wants_os_action("refresh the page"))
        self.assertTrue(wants_os_action("help center kholo"))
        # bilingual: Hindi, English, Hinglish, either script — same meaning
        self.assertTrue(wants_os_action("open browser"))
        self.assertTrue(wants_os_action("browser open karo"))
        self.assertTrue(wants_os_action("ओपन ब्राउज़र"))
        self.assertTrue(wants_os_action("browser kholo"))
        self.assertTrue(wants_os_action("clear the board"))
        self.assertTrue(wants_os_action("board saaf karo"))
        self.assertTrue(wants_os_action("open whiteboard"))
        self.assertTrue(wants_os_action("notes open karo"))
        # multilingual fast path: es/fr/de/pt + ta/te/kn/ml/bn/mr
        self.assertTrue(wants_os_action("abre el navegador"))
        self.assertTrue(wants_os_action("ouvre le navigateur"))
        self.assertTrue(wants_os_action("öffne den Browser"))
        self.assertTrue(wants_os_action("abrir nova aba"))
        self.assertTrue(wants_os_action("உலாவியை திற"))
        self.assertTrue(wants_os_action("బ్రౌజర్ తెరువు"))
        self.assertTrue(wants_os_action("ಬ್ರೌಸರ್ ತೆರೆ"))
        self.assertTrue(wants_os_action("ബ്രൗസർ തുറക്കൂ"))
        self.assertTrue(wants_os_action("ব্রাউজার খুলুন"))
        self.assertTrue(wants_os_action("ब्राउझर उघड"))
        # precision: substrings inside other words must NOT trigger
        self.assertFalse(wants_os_action("this is notable"))
        self.assertFalse(wants_os_action("about everything"))
        self.assertFalse(wants_os_action("browser kya hai?"))
        self.assertFalse(wants_os_action("ye kya hai"))
        self.assertFalse(wants_os_action("hi"))

    def test_ack_block(self):
        acts = [
            {"op": "open_app", "app": "browser"},
            {"op": "browser_navigate", "target": "python docs"},
        ]
        block = ack_block(acts)
        self.assertIn("opened the Browser", block)
        self.assertIn("python docs", block)
        self.assertIn("Devanagari", block)
        honest = ack_block([])
        self.assertIn("none", honest)
        self.assertIn("Do NOT claim", honest)
        self.assertEqual(
            describe_action({"op": "tile_app", "app": "code", "zone": "tl"}),
            "tiled the Code editor to the top-left quarter",
        )
        self.assertEqual(describe_action({"op": "browser_reload"}), "reloaded the browser page")
        self.assertEqual(describe_action({"op": "browser_close_tab"}), "closed the browser tab")


if __name__ == "__main__":
    unittest.main()
