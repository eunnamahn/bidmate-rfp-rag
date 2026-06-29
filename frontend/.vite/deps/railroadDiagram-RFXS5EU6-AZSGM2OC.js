import {
  db,
  getStyles,
  renderer
} from "./chunk-2W2Z3FHT.js";
import {
  populateCommonDb
} from "./chunk-LYOQ6TCY.js";
import {
  MermaidParseError
} from "./chunk-2SPSUN4U.js";
import "./chunk-VFX2UPZ3.js";
import "./chunk-ZH7SVVH7.js";
import "./chunk-3SX2RRW4.js";
import "./chunk-JF7M3FU3.js";
import "./chunk-2VBS5NCY.js";
import "./chunk-ROFPAOJC.js";
import "./chunk-UTS3S3JO.js";
import {
  createRailroadServices
} from "./chunk-FWXVORCY.js";
import "./chunk-S5WWHIRC.js";
import "./chunk-IG4SYBEJ.js";
import "./chunk-5SSAL7U3.js";
import "./chunk-XGDJSZVJ.js";
import "./chunk-DQU7WUGC.js";
import "./chunk-RAMB7O63.js";
import "./chunk-KYEZTJNO.js";
import "./chunk-3L74IPNC.js";
import "./chunk-AZSUYZZX.js";
import "./chunk-CGQCCDEI.js";
import {
  log
} from "./chunk-GVR2T37F.js";
import {
  __name
} from "./chunk-UEB47VSI.js";
import "./chunk-WOOG5QLI.js";

// node_modules/mermaid/dist/chunks/mermaid.core/railroadDiagram-RFXS5EU6.mjs
var langiumParser = createRailroadServices().Railroad.parser.LangiumParser;
var transformExpression = __name((expr) => {
  switch (expr.$type) {
    case "RailroadTerminalExpr":
      return {
        type: "terminal",
        value: expr.value
      };
    case "RailroadNonTerminalExpr":
      return {
        type: "nonterminal",
        name: expr.name
      };
    case "RailroadSpecialExpr":
      return {
        type: "special",
        text: expr.text
      };
    case "RailroadSequenceExpr": {
      const elements = expr.elements.map(transformExpression);
      return elements.length === 1 ? elements[0] : { type: "sequence", elements };
    }
    case "RailroadChoiceExpr": {
      const alternatives = expr.alternatives.map(transformExpression);
      return alternatives.length === 1 ? alternatives[0] : { type: "choice", alternatives };
    }
    case "RailroadOptionalExpr":
      return {
        type: "optional",
        element: transformExpression(expr.element)
      };
    case "RailroadOneOrMoreExpr":
      return {
        type: "repetition",
        element: transformExpression(expr.element),
        min: 1,
        max: Infinity
      };
    case "RailroadZeroOrMoreExpr":
      return {
        type: "repetition",
        element: transformExpression(expr.element),
        min: 0,
        max: Infinity
      };
    default:
      throw new Error(`Unsupported railroad expression: ${expr.$type}`);
  }
}, "transformExpression");
var transformRule = __name((rule) => {
  return {
    name: rule.name,
    definition: transformExpression(rule.definition)
  };
}, "transformRule");
var populateDb = __name((ast) => {
  populateCommonDb(ast, db);
  if (ast.title) {
    db.setTitle(ast.title);
  }
  ast.rules.map((rule) => db.addRule(transformRule(rule)));
}, "populateDb");
var parser = {
  parse: __name((input) => {
    db.clear();
    log.debug("[Railroad Parser] Starting Langium parse");
    const result = langiumParser.parse(input);
    if (result.lexerErrors.length > 0 || result.parserErrors.length > 0) {
      throw new MermaidParseError(result);
    }
    const ast = result.value;
    log.debug("[Railroad Parser] Parsed rules:", ast.rules.length);
    populateDb(ast);
    log.debug("[Railroad Parser] Parse complete");
  }, "parse"),
  parser: {
    yy: db
  }
};
var diagram = {
  parser,
  db,
  renderer,
  styles: getStyles
};
var railroadDiagram_default = diagram;
export {
  railroadDiagram_default as default,
  diagram
};
//# sourceMappingURL=railroadDiagram-RFXS5EU6-AZSGM2OC.js.map
