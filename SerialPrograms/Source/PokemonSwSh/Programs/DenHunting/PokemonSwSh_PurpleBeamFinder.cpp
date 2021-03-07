/*  Purple Beam Finder
 *
 *  From: https://github.com/PokemonAutomation/Arduino-Source
 *
 */

#include "Common/SwitchFramework/FrameworkSettings.h"
#include "Common/SwitchFramework/Switch_PushButtons.h"
#include "Common/PokemonSwSh/PokemonSettings.h"
#include "Common/PokemonSwSh/PokemonSwShGameEntry.h"
#include "PokemonSwSh/Inference/PokemonSwSh_BeamSetter.h"
#include "PokemonSwSh_PurpleBeamFinder.h"

namespace PokemonAutomation{
namespace NintendoSwitch{
namespace PokemonSwSh{


PurpleBeamFinder::PurpleBeamFinder()
    : SingleSwitchProgram(
        FeedbackType::REQUIRED, PABotBaseLevel::PABOTBASE_12KB,
        "Purple Beam Finder",
        "NativePrograms/PurpleBeamFinder.md",
        "Automatically reset for a purple beam."
    )
    , EXTRA_LINE(
        "<b>Extra Line:</b><br>(German has an extra line of text.)",
        false
    )
    , DETECTION_THRESHOLD(
        "<b>Red Beam Detection Threshold:</b>",
        0.01, 0.0, 1.0
    )
    , TIMEOUT_DELAY(
        "<b>Timeout Delay:</b><br>Reset if no beam is detected after this long.",
        "2 * TICKS_PER_SECOND"
    )
{
    m_options.emplace_back(&EXTRA_LINE, "EXTRA_LINE");
    m_options.emplace_back(&DETECTION_THRESHOLD, "DETECTION_THRESHOLD");
    m_options.emplace_back(&TIMEOUT_DELAY, "TIMEOUT_DELAY");
}

void PurpleBeamFinder::program(SingleSwitchProgramEnvironment& env) const{
    grip_menu_connect_go_home();

    resume_game_front_of_den_nowatts(TOLERATE_SYSTEM_UPDATE_MENU_SLOW);
    pbf_mash_button(BUTTON_B, 100);
    env.console.botbase().wait_for_all_requests();

    BeamSetter setter(env.console, env.logger);

    uint64_t count = 0;
    while (true){
        env.logger.log("Attempts: " + QString::number(count));

        //  Talk to den.
        pbf_press_button(BUTTON_A, 10, 450);
        if (EXTRA_LINE){
            pbf_press_button(BUTTON_A, 10, 300);
        }
        pbf_press_button(BUTTON_A, 10, 300);
        env.console.botbase().wait_for_all_requests();

        if (setter.run(env, env.console, DETECTION_THRESHOLD, TIMEOUT_DELAY)){
            break;
        }

        pbf_press_button(BUTTON_HOME, 10, GAME_TO_HOME_DELAY_SAFE);
        reset_game_from_home(TOLERATE_SYSTEM_UPDATE_MENU_SLOW, 0, false);

        count++;
    }

    while (true){
        pbf_press_button(BUTTON_B, 5, 5);
        pbf_press_button(BUTTON_LCLICK, 5, 5);
    }
}





}
}
}
