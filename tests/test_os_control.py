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

    def test_action_gate(self):
        self.assertTrue(wants_os_action("browser kholo"))
        self.assertTrue(wants_os_action("notes me ye likh do"))
        self.assertTrue(wants_os_action("python docs dikhao"))
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


if __name__ == "__main__":
    unittest.main()
