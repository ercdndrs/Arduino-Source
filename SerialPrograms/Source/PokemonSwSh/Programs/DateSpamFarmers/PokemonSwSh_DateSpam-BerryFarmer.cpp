/*  Berry Farmer
 *
 *  From: https://github.com/PokemonAutomation/Arduino-Source
 *
 */

#include "Common/Clientside/PrettyPrint.h"
#include "Common/SwitchFramework/Switch_PushButtons.h"
#include "Common/PokemonSwSh/PokemonSettings.h"
#include "Common/PokemonSwSh/PokemonSwShGameEntry.h"
#include "Common/PokemonSwSh/PokemonSwShDateSpam.h"
#include "PokemonSwSh_DateSpam-BerryFarmer.h"

namespace PokemonAutomation{
namespace NintendoSwitch{
namespace PokemonSwSh{

BerryFarmer::BerryFarmer()
    : SingleSwitchProgram(
        FeedbackType::NONE, PABotBaseLevel::PABOTBASE_12KB,
        "Date Spam: Berry Farmer",
        "NativePrograms/DateSpam-BerryFarmer.md",
        "Farm berries."
    )
    , SKIPS(
        "<b>Number of Fetch Attempts:</b>",
        100000
    )
    , SAVE_ITERATIONS(
        "<b>Save Every this Many Fetches:</b><br>(zero disables saving): ",
        0
    )
{
    m_options.emplace_back(&SKIPS, "SKIPS");
    m_options.emplace_back(&SAVE_ITERATIONS, "SAVE_ITERATIONS");
}

void BerryFarmer::program(SingleSwitchProgramEnvironment& env) const{
    grip_menu_connect_go_home(env.console);

    uint8_t year = MAX_YEAR;
    uint16_t save_count = 0;
    for (uint32_t c = 0; c < SKIPS; c++){
        env.log("Fetch Attempts: " + tostr_u_commas(c));

        home_roll_date_enter_game_autorollback(env.console, &year);
        pbf_mash_button(env.console, BUTTON_B, 90);

        pbf_press_button(env.console, BUTTON_A, 10, 10);
        pbf_mash_button(env.console, BUTTON_ZL, 385);
        pbf_mash_button(env.console, BUTTON_B, 600);

        if (SAVE_ITERATIONS != 0){
            save_count++;
            if (save_count >= SAVE_ITERATIONS){
                save_count = 0;
                pbf_mash_button(env.console, BUTTON_B, 2 * TICKS_PER_SECOND);
                pbf_press_button(env.console, BUTTON_X, 20, OVERWORLD_TO_MENU_DELAY);
                pbf_press_button(env.console, BUTTON_R, 20, 2 * TICKS_PER_SECOND);
                pbf_press_button(env.console, BUTTON_ZL, 20, 3 * TICKS_PER_SECOND);
            }
        }

        //  Tap HOME and quickly spam B. The B spamming ensures that we don't
        //  accidentally update the system if the system update window pops up.
        pbf_press_button(env.console, BUTTON_HOME, 10, 5);
        pbf_mash_button(env.console, BUTTON_B, GAME_TO_HOME_DELAY_FAST - 15);
    }

    end_program_callback(env.console);
    end_program_loop(env.console);
}




}
}
}
