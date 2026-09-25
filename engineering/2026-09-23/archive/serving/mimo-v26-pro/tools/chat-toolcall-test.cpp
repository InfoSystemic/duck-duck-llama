// chat-toolcall-test: render MiMo-V2.6-Pro's own chat template with tools through llama.cpp's chat layer, then parse
// candidate model outputs exactly as llama-server does (common_chat_parse with the serialized PEG parser), and print
// what the client would receive. Offline: no model, no server restart.
//
// usage: chat-toolcall-test <chat_template.jinja> [--grammar]
#include "chat.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <chat_template.jinja> [--grammar]\n", argv[0]);
        return 1;
    }
    std::ifstream f(argv[1]);
    std::stringstream ss; ss << f.rdbuf();
    const bool show_grammar = argc > 2 && std::string(argv[2]) == "--grammar";

    auto tmpls = common_chat_templates_init(nullptr, ss.str(), "", "<|im_end|>");

    common_chat_templates_inputs in;
    common_chat_msg user;
    user.role = "user";
    user.content = "What's the weather in Paris right now? Use the tool.";
    in.messages = { user };
    common_chat_tool weather{ "get_weather", "Get the current weather for a city.",
        R"({"type":"object","properties":{"city":{"type":"string","description":"City name"},"days":{"type":"integer"}},"required":["city"]})" };
    common_chat_tool edit{ "edit_file", "Replace text in a file.",
        R"({"type":"object","properties":{"path":{"type":"string"},"old":{"type":"string"},"new":{"type":"string"}},"required":["path","old","new"]})" };
    in.tools = { weather, edit };
    in.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    in.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

    common_chat_params cp = common_chat_templates_apply(tmpls.get(), in);
    printf("format: %s  grammar_lazy: %d  triggers: %zu  generation_prompt: %s\n", common_chat_format_name(cp.format),
           (int) cp.grammar_lazy, cp.grammar_triggers.size(), cp.generation_prompt.c_str());
    if (show_grammar) {
        printf("---- grammar ----\n%s\n-----------------\n", cp.grammar.c_str());
    }

    common_chat_parser_params pp(cp);
    pp.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    pp.parser.load(cp.parser);

    const std::vector<std::pair<std::string, std::string>> cases = {
        { "compact (MiMo's own template form)",
          "<think>User wants weather.</think><tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>" },
        { "newline layout (Qwen3-Coder form)",
          "<think>User wants weather.</think><tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>" },
        { "compact, two params incl. integer",
          "<think>x</think><tool_call><function=get_weather><parameter=city>Paris</parameter><parameter=days>3</parameter></function></tool_call>" },
        { "compact, multi-line string values",
          "<think>x</think><tool_call><function=edit_file><parameter=path>a.py</parameter><parameter=old>def f():\n    return 1</parameter><parameter=new>def f():\n    return 2\n</parameter></function></tool_call>" },
        { "observed garbage (</invoke> in value)",
          "<think>x</think><tool_call><function=get_weather><parameter=city>Paris\n</invoke>\n</parameter>\n</function>\n</tool_call>" },
        { "text content then compact call",
          "<think>x</think>Let me check.<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>" },
    };
    for (const auto & [label, out] : cases) {
        try {
            common_chat_msg msg = common_chat_parse(cp.generation_prompt + out, false, pp);
            printf("\n[%s]\n  content=%s\n", label.c_str(), msg.content.c_str());
            if (msg.tool_calls.empty()) {
                printf("  NO TOOL CALLS PARSED\n");
            }
            for (const auto & tc : msg.tool_calls) {
                printf("  tool_call name=%s args=%s\n", tc.name.c_str(), tc.arguments.c_str());
            }
        } catch (const std::exception & e) {
            printf("\n[%s]\n  PARSE EXCEPTION: %s\n", label.c_str(), e.what());
        }
    }
    return 0;
}
