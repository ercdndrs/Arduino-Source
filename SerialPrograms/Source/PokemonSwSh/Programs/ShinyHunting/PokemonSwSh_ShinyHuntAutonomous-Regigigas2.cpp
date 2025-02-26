/*  Shiny Hunt Autonomous - Regigigas2
 *
 *  From: https://github.com/PokemonAutomation/Arduino-Source
 *
 */

#include "Common/Clientside/PrettyPrint.h"
#include "Common/SwitchFramework/FrameworkSettings.h"
#include "Common/SwitchFramework/Switch_PushButtons.h"
#include "Common/PokemonSwSh/PokemonSettings.h"
#include "Common/PokemonSwSh/PokemonSwShGameEntry.h"
#include "Common/PokemonSwSh/PokemonSwShDateSpam.h"
#include "CommonFramework/PersistentSettings.h"
#include "PokemonSwSh/Inference/PokemonSwSh_StartBattleDetector.h"
#include "PokemonSwSh/Inference/PokemonSwSh_RaidCatchDetector.h"
#include "PokemonSwSh/Inference/ShinyDetection/PokemonSwSh_ShinyEncounterDetector.h"
#include "PokemonSwSh/Programs/PokemonSwSh_StartGame.h"
#include "PokemonSwSh_ShinyHuntAutonomous-Regigigas2.h"

namespace PokemonAutomation{
namespace NintendoSwitch{
namespace PokemonSwSh{

ShinyHuntAutonomousRegigigas2::ShinyHuntAutonomousRegigigas2()
    : SingleSwitchProgram(
        FeedbackType::REQUIRED, PABotBaseLevel::PABOTBASE_12KB,
        "Shiny Hunt Autonomous - Regigigas2",
        "SerialPrograms/ShinyHuntAutonomous-Regigigas2.md",
        "Automatically hunt for shiny Regigigas using video feedback."
    )
    , GO_HOME_WHEN_DONE(
        "<b>Go Home when Done:</b><br>After finding a shiny, go to the Switch Home menu to idle. (turn this off for unattended streaming)",
        false
    )
    , REVERSAL_PP(
        "<b>Reversal PP:</b><br>The amount of Reversal PP you are saved with.",
        24
    )
    , REQUIRE_SQUARE(
        "<b>Require Square:</b><br>Stop only for a square shiny. Run from star shinies.",
        false
    )
    , TOUCH_DATE_INTERVAL(
        "<b>Rollover Prevention:</b><br>Prevent a den from rolling over by periodically touching the date. If set to zero, this feature is disabled.",
        "4 * 3600 * TICKS_PER_SECOND"
    )
    , m_advanced_options(
        "<font size=4><b>Advanced Options:</b> You should not need to touch anything below here.</font>"
    )
    , CATCH_TO_OVERWORLD_DELAY(
        "<b>Catch to Overworld Delay:</b>",
        "8 * TICKS_PER_SECOND"
    )
    , VIDEO_ON_SHINY(
        "<b>Video Capture:</b><br>Take a video of the encounter if it is shiny.",
        true
    )
    , RUN_FROM_EVERYTHING(
        "<b>Run from Everything:</b><br>Run from everything - even if it is shiny. (For testing only.)",
        false
    )
{
    m_options.emplace_back(&GO_HOME_WHEN_DONE, "GO_HOME_WHEN_DONE");
    m_options.emplace_back(&REVERSAL_PP, "REVERSAL_PP");
    m_options.emplace_back(&REQUIRE_SQUARE, "REQUIRE_SQUARE");
    m_options.emplace_back(&TOUCH_DATE_INTERVAL, "TOUCH_DATE_INTERVAL");
    m_options.emplace_back(&m_advanced_options, "");
    m_options.emplace_back(&CATCH_TO_OVERWORLD_DELAY, "CATCH_TO_OVERWORLD_DELAY");
    if (settings.developer_mode){
        m_options.emplace_back(&VIDEO_ON_SHINY, "VIDEO_ON_SHINY");
        m_options.emplace_back(&RUN_FROM_EVERYTHING, "RUN_FROM_EVERYTHING");
    }
}




struct ShinyHuntAutonomousRegigigas2::Stats : public ShinyHuntTracker{
    Stats()
        : ShinyHuntTracker(true)
        , m_timeouts(m_stats["Timeouts"])
    {
        m_display_order.insert(m_display_order.begin() + 1, Stat("Timeouts"));
    }
    uint64_t& m_timeouts;
};
std::unique_ptr<StatsTracker> ShinyHuntAutonomousRegigigas2::make_stats() const{
    return std::unique_ptr<StatsTracker>(new Stats());
}




ShinyHuntAutonomousRegigigas2::Tracker::Tracker(
    ShinyHuntTracker& stats,
    ProgramEnvironment& env,
    ConsoleHandle& console,
    bool require_square,
    uint16_t exit_battle_time,
    bool take_video,
    bool run_from_everything
)
    : StandardEncounterTracker(
        stats, env, console,
        require_square,
        exit_battle_time,
        take_video, run_from_everything
    )
    , m_env(env)
{}
bool ShinyHuntAutonomousRegigigas2::Tracker::run_away(){
    RaidCatchDetector detector(m_console, std::chrono::seconds(30));
    pbf_mash_button(m_console, BUTTON_A, 4 * TICKS_PER_SECOND);

    if (!detector.wait(m_env)){
        m_env.log("Raid Catch Menu not found.", Qt::red);
        return false;
    }

    pbf_press_dpad(m_console, DPAD_DOWN, 10, 0);
    pbf_press_button(m_console, BUTTON_A, 10, m_exit_battle_time);
    return true;
}

bool ShinyHuntAutonomousRegigigas2::kill_and_return(SingleSwitchProgramEnvironment& env) const{
    RaidCatchDetector detector(env.console, std::chrono::seconds(30));
    pbf_mash_button(env.console, BUTTON_A, 4 * TICKS_PER_SECOND);

    if (!detector.wait(env)){
        env.log("Raid Catch Menu not found.", Qt::red);
        return false;
    }

    pbf_press_dpad(env.console, DPAD_DOWN, 10, 0);
    pbf_press_button(env.console, BUTTON_A, 10, CATCH_TO_OVERWORLD_DELAY);
    return true;
}
void ShinyHuntAutonomousRegigigas2::program(SingleSwitchProgramEnvironment& env) const{
    grip_menu_connect_go_home(env.console);

    uint32_t last_touch = system_clock(env.console);
    if (TOUCH_DATE_INTERVAL > 0){
        touch_date_from_home(env.console, SETTINGS_TO_HOME_DELAY);
    }

    resume_game_back_out(env.console, TOLERATE_SYSTEM_UPDATE_MENU_FAST, 500);

    Stats& stats = env.stats<Stats>();
    Tracker tracker(
        stats,
        env, env.console,
        REQUIRE_SQUARE,
        0,
        VIDEO_ON_SHINY,
        RUN_FROM_EVERYTHING
    );

    while (true){
        for (uint8_t pp = REVERSAL_PP; pp > 0; pp--){
            env.update_stats();

            env.log("Starting Regigigas Encounter: " + tostr_u_commas(stats.encounters() + 1));

            pbf_mash_button(env.console, BUTTON_A, 18 * TICKS_PER_SECOND);
            env.console.botbase().wait_for_all_requests();

            if (!wait_for_start_battle(env, env.console, std::chrono::seconds(30))){
                stats.m_timeouts++;
                break;
            }

            ShinyDetection detection = detect_shiny_battle(
                env, env.console,
                SHINY_BATTLE_RAID,
                std::chrono::seconds(30)
            );

            if (tracker.process_result(detection)){
                goto StopProgram;
            }
            if (detection == ShinyDetection::NO_BATTLE_MENU || !tracker.run_away()){
                stats.m_timeouts++;
                break;
            }
        }

        pbf_press_button(env.console, BUTTON_HOME, 10, GAME_TO_HOME_DELAY_SAFE);
        if (TOUCH_DATE_INTERVAL > 0 && system_clock(env.console) - last_touch >= TOUCH_DATE_INTERVAL){
            touch_date_from_home(env.console, SETTINGS_TO_HOME_DELAY);
            last_touch += TOUCH_DATE_INTERVAL;
        }
        reset_game_from_home_with_inference(
            env, env.console,
            TOLERATE_SYSTEM_UPDATE_MENU_FAST
        );
    }


StopProgram:
    env.update_stats();

    if (GO_HOME_WHEN_DONE){
        pbf_press_button(env.console, BUTTON_HOME, 10, GAME_TO_HOME_DELAY_SAFE);
    }

    end_program_callback(env.console);
    end_program_loop(env.console);
}



}
}
}


