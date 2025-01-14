/*  ShinyHuntUnattended-IoATrade
 *
 *  From: https://github.com/PokemonAutomation/Arduino-Source
 *
 */

#ifndef PokemonAutomation_PokemonSwSh_ShinyHuntUnattendedIoATrade_H
#define PokemonAutomation_PokemonSwSh_ShinyHuntUnattendedIoATrade_H

#include "CommonFramework/Options/SectionDivider.h"
#include "NintendoSwitch/Options/TimeExpression.h"
#include "NintendoSwitch/Framework/SingleSwitchProgram.h"

namespace PokemonAutomation{
namespace NintendoSwitch{
namespace PokemonSwSh{

class ShinyHuntUnattendedIoATrade : public SingleSwitchProgram{
public:
    ShinyHuntUnattendedIoATrade();

    virtual void program(SingleSwitchProgramEnvironment& env) const override;

private:
    TimeExpression<uint16_t> START_TO_RUN_DELAY;
    TimeExpression<uint32_t> TOUCH_DATE_INTERVAL;
    SectionDivider m_advanced_options;
    TimeExpression<uint16_t> FLY_DURATION;
    TimeExpression<uint16_t> MOVE_DURATION;
    TimeExpression<uint16_t> MASH_TO_TRADE_DELAY;
};


}
}
}
#endif
