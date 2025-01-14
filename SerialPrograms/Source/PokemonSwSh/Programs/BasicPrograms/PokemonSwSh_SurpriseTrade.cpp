/*  Surprise Trade
 *
 *  From: https://github.com/PokemonAutomation/Arduino-Source
 *
 */

#include "Common/SwitchFramework/FrameworkSettings.h"
#include "Common/SwitchFramework/Switch_PushButtons.h"
#include "Common/PokemonSwSh/PokemonSettings.h"
#include "Common/PokemonSwSh/PokemonSwShGameEntry.h"
#include "NintendoSwitch/FixedInterval.h"
#include "PokemonSwSh_SurpriseTrade.h"

namespace PokemonAutomation{
namespace NintendoSwitch{
namespace PokemonSwSh{

SurpriseTrade::SurpriseTrade()
    : SingleSwitchProgram(
        FeedbackType::NONE, PABotBaseLevel::PABOTBASE_12KB,
        "Surprise Trade",
        "NativePrograms/SurpriseTrade.md",
        "Surprise trade away boxes of " + STRING_POKEMON
    )
    , BOXES_TO_TRADE(
        "<b>Number of Boxes to Trade:</b>",
        2
    )
    , INITIAL_WAIT(
        "<b>Time to wait for a Trade Partner:</b>",
        "30 * TICKS_PER_SECOND"
    )
    , m_advanced_options(
        "<font size=4><b>Advanced Options:</b> You should not need to touch anything below here.</font>"
    )
    , TRADE_ANIMATION(
        "<b>Trade Animation Time:</b>",
        "23 * TICKS_PER_SECOND"
    )
    , EVOLVE_DELAY(
        "<b>Evolve Delay:</b>",
        "30 * TICKS_PER_SECOND"
    )
{
    m_options.emplace_back(&BOXES_TO_TRADE, "BOXES_TO_TRADE");
    m_options.emplace_back(&INITIAL_WAIT, "INITIAL_WAIT");
    m_options.emplace_back(&m_advanced_options, "");
    m_options.emplace_back(&TRADE_ANIMATION, "TRADE_ANIMATION");
    m_options.emplace_back(&EVOLVE_DELAY, "EVOLVE_DELAY");
}


void SurpriseTrade::trade_slot(const BotBaseContext& context, uint8_t slot, bool next_box) const{
    ssf_press_button2(context, BUTTON_Y, OPEN_YCOMM_DELAY, 50);
    ssf_press_dpad1(context, DPAD_DOWN, 10);
    ssf_press_button2(context, BUTTON_A, 280, 20);

    if (next_box){
        ssf_press_button1(context, BUTTON_R, BOX_CHANGE_DELAY);
    }

    //  Move to slot
    while (slot >= 6){
        ssf_press_dpad1(context, DPAD_DOWN, BOX_SCROLL_DELAY);
        slot -= 6;
    }
    while (slot > 0){
        ssf_press_dpad1(context, DPAD_RIGHT, BOX_SCROLL_DELAY);
        slot--;
    }

    ssf_press_button1(context, BUTTON_A, 50);
    ssf_press_button1(context, BUTTON_A, 500);
    ssf_press_button1(context, BUTTON_A, 100);
    ssf_press_button1(context, BUTTON_A, 100);

    pbf_mash_button(context, BUTTON_B, INITIAL_WAIT);

    //  This is a state-merging operation.
    //  If we just finished a trade, this will start the animation for it.
    //  If we failed the previous trade and are stuck in the wrong parity, this
    //  is a no-op that will correct the parity and setup the next trade.
    ssf_press_button1(context, BUTTON_Y, OPEN_YCOMM_DELAY);
    pbf_mash_button(context, BUTTON_B, TRADE_ANIMATION);
}

void SurpriseTrade::program(SingleSwitchProgramEnvironment& env) const{
    grip_menu_connect_go_home(env.console);
    resume_game_no_interact(env.console, TOLERATE_SYSTEM_UPDATE_MENU_FAST);

    for (uint8_t box = 0; box < BOXES_TO_TRADE; box++){
        //  At this point, we MUST be in the overworld with no pending trade.
        //  Otherwise the box transition will fail. Therefore we add a cleanup
        //  stage after each box to make sure we are in this state.
        trade_slot(env.console, 0, box != 0);

        for (uint8_t c = 1; c < 30; c++){
            trade_slot(env.console, c, false);
        }

        //  If the previous trade isn't done, either wait to finish or cancel it.
        //  Two iterations of this will merge all the following states:
        //      1.  The last trade finished successfully. Both iterations are no-ops.
        //      2.  The last trade was slightly too slow. The 1st iteration
        //          will finish it. The 2nd iteration is a no-op.
        //      3.  The last trade was very slow. The 1st iteration is suppressed
        //          because the trade is in progress. The 2nd iteration finishes it.
        //      4.  No partner was ever found. The 1st iteration will cancel the trade.
        for (uint8_t c = 0; c < 2; c++){
            ssf_press_button1(env.console, BUTTON_Y, 250);
            ssf_press_dpad1(env.console, DPAD_DOWN, 20);
            ssf_press_button1(env.console, BUTTON_A, 280);
            ssf_press_button1(env.console, BUTTON_B, 280);
            ssf_press_button1(env.console, BUTTON_B, 200);
            ssf_press_button1(env.console, BUTTON_A, 100);
            pbf_mash_button(env.console, BUTTON_B, TRADE_ANIMATION);
        }

        //  Wait out any new pokedex entries or trade evolutions.
        pbf_mash_button(env.console, BUTTON_B, EVOLVE_DELAY);
    }

    ssf_press_button2(env.console, BUTTON_HOME, GAME_TO_HOME_DELAY_SAFE, 10);

    end_program_callback(env.console);
    end_program_loop(env.console);
}


}
}
}
