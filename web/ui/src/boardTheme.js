/* Single source of truth for the tutor board's look.
   Values mirror the app shell (App.jsx / index.css) so the board always
   feels like the same product: coral primary, deep-sea ink, teal secondary.
   Change the theme HERE — never hardcode colors in TutorBoard.jsx. */

export const BOARD_THEME = {
  primary: "#ff5a5f", // coral — arrows, active step, pen, accents
  primarySoft: "#fff1f1", // shape fill tint
  ink: "#12304a", // node strokes, body text
  inkSoft: "#34566b", // secondary strokes
  teal: "#0f766e", // secondary accents, note callouts
  tealSoft: "#ccfbf1", // note background tint
  paper: "#ffffff", // board background
  line: "#e2e8f0", // slate-200 hairlines
  muted: "#64748b", // slate-500 secondary text
  faint: "#94a3b8", // slate-400 placeholders
  codeBg: "#0f172a", // slate-900 code cards
  codeInk: "#e2e8f0", // code text
  radius: 12, // px — compact professional nodes
  // Semantic tones: the planner tags core concept / example / warning.
  tones: {
    core: { fill: "#fff1f1", stroke: "#ff5a5f" },
    example: { fill: "#ecfdf5", stroke: "#0f766e" },
    warn: { fill: "#fffbeb", stroke: "#d97706" },
  },
};
