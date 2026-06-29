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
import "./chunk-FWXVORCY.js";
import "./chunk-S5WWHIRC.js";
import "./chunk-IG4SYBEJ.js";
import {
  createRailroadPegServices
} from "./chunk-5SSAL7U3.js";
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

// node_modules/mermaid/dist/chunks/mermaid.core/pegDiagram-2B236MQR.mjs
var langiumParser = createRailroadPegServices().RailroadPeg.parser.LangiumParser;
var transformOrderedChoice = __name((choice) => {
  const alternatives = choice.alternatives.map(transformSequence);
  if (alternatives.length === 1) {
    return alternatives[0];
  }
  return {
    type: "choice",
    alternatives
  };
}, "transformOrderedChoice");
var transformSequence = __name((sequence) => {
  const elements = sequence.elements.map(transformPrefix);
  if (elements.length === 1) {
    return elements[0];
  }
  return {
    type: "sequence",
    elements
  };
}, "transformSequence");
var transformPrefix = __name((prefix) => {
  const inner = transformSuffix(prefix.suffix);
  if (!prefix.operator) {
    return inner;
  }
  const label = prefix.operator === "&" ? `&${nodeToLabel(inner)}` : `!${nodeToLabel(inner)}`;
  return {
    type: "special",
    text: label
  };
}, "transformPrefix");
var nodeToLabel = __name((node) => {
  switch (node.type) {
    case "terminal":
      return `"${node.value}"`;
    case "nonterminal":
      return node.name;
    case "special":
      return node.text;
    default:
      return "(...)";
  }
}, "nodeToLabel");
var transformSuffix = __name((suffix) => {
  const inner = transformPrimary(suffix.primary);
  if (!suffix.operator) {
    return inner;
  }
  switch (suffix.operator) {
    case "?":
      return { type: "optional", element: inner };
    case "*":
      return { type: "repetition", element: inner, min: 0, max: Infinity };
    case "+":
      return { type: "repetition", element: inner, min: 1, max: Infinity };
    default:
      throw new Error(`Unsupported PEG suffix operator: ${suffix.operator}`);
  }
}, "transformSuffix");
var transformPrimary = __name((primary) => {
  switch (primary.$type) {
    case "PegLiteral":
      return {
        type: "terminal",
        value: primary.value
      };
    case "PegIdentifier":
      return {
        type: "nonterminal",
        name: primary.name
      };
    case "PegGroup":
      return transformOrderedChoice(primary.element);
    case "PegAny":
      return {
        type: "special",
        text: primary.dot
      };
    default:
      throw new Error(`Unsupported PEG primary node: ${primary.$type}`);
  }
}, "transformPrimary");
var transformRule = __name((rule) => {
  return {
    name: rule.name,
    definition: transformOrderedChoice(rule.definition)
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
    log.debug("[PEG Parser] Starting Langium parse");
    const result = langiumParser.parse(input);
    if (result.lexerErrors.length > 0 || result.parserErrors.length > 0) {
      throw new MermaidParseError(result);
    }
    const ast = result.value;
    log.debug("[PEG Parser] Parsed rules:", ast.rules.length);
    populateDb(ast);
    log.debug("[PEG Parser] Parse complete");
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
export {
  diagram
};
//# sourceMappingURL=pegDiagram-2B236MQR-RZH4YY2G.js.map
