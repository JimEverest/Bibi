import unittest

from app.floating_button import (
    FloatingButton,
    FloatingButtonController,
    FloatingButtonDragController,
    _BAR_BG,
    _COLLAPSED_SIZE,
)


class FloatingButtonControllerTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.controller = FloatingButtonController(
            start_recording=lambda: self.events.append("start"),
            stop_recording=lambda: self.events.append("stop"),
            is_recording=lambda: self.events.count("start") > self.events.count("stop"),
        )

    def test_main_area_press_and_release_records_once(self):
        self.controller.press_main()
        self.controller.press_main()
        self.controller.release_main()
        self.controller.release_main()

        self.assertEqual(self.events, ["start", "stop"])

    def test_embedded_button_click_toggles_recording(self):
        self.controller.click_toggle()
        self.controller.click_toggle()

        self.assertEqual(self.events, ["start", "stop"])

    def test_main_area_release_does_not_stop_other_recording(self):
        self.controller.click_toggle()
        self.controller.press_main()
        self.controller.release_main()

        self.assertEqual(self.events, ["start"])


class FloatingButtonAppearanceTests(unittest.TestCase):
    def test_bar_uses_requested_blue_and_expanded_collapsed_size(self):
        self.assertEqual(_BAR_BG, "#0473ea")
        self.assertEqual(_COLLAPSED_SIZE, (68, 16))

    def test_right_click_dispatches_to_context_menu_callback(self):
        button = FloatingButton.__new__(FloatingButton)
        button._root = object()
        events = []
        button.set_context_menu_callback(
            lambda root, x, y: events.append((root, x, y))
        )
        event = type("Event", (), {"x_root": 321, "y_root": 654})()

        result = button._on_context_menu(event)

        self.assertEqual(result, "break")
        self.assertEqual(events, [(button._root, 321, 654)])


class FloatingButtonDragControllerTests(unittest.TestCase):
    def test_bar_click_requests_collapse_toggle(self):
        controller = FloatingButtonDragController()

        controller.press(100, 200, 20, 30)

        self.assertTrue(controller.release())

    def test_drag_moves_window_without_collapse_toggle(self):
        controller = FloatingButtonDragController()
        controller.press(100, 200, 20, 30)

        self.assertEqual(controller.move(112, 225), (32, 55))
        self.assertFalse(controller.release())


if __name__ == "__main__":
    unittest.main()
