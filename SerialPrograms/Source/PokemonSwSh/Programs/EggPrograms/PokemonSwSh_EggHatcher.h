/*  Egg Hatcher
 *
 *  From: https://github.com/PokemonAutomation/Arduino-Source
 *
 */

#ifndef PokemonAutomation_PokemonSwSh_EggHatcher_H
#define PokemonAutomation_PokemonSwSh_EggHatcher_H

#include "CommonFramework/Options/SectionDivider.h"
#include "CommonFramework/Options/SimpleInteger.h"
#include "NintendoSwitch/Options/TimeExpression.h"
#include "NintendoSwitch/Framework/SingleSwitchProgram.h"
#include "PokemonSwSh/Options/EggStepCount.h"
#include "PokemonSwSh_EggHelpers.h"

namespace PokemonAutomation{
namespace NintendoSwitch{
namespace PokemonSwSh{


class EggHatcher : public SingleSwitchProgram{
public:
    EggHatcher();

    virtual void program(SingleSwitchProgramEnvironment& env) const override;


private:
    SimpleInteger<uint8_t> BOXES_TO_HATCH;
    EggStepCount STEPS_TO_HATCH;
    SectionDivider m_advanced_options;
    TimeExpression<uint16_t> SAFETY_TIME;
    TimeExpression<uint16_t> HATCH_DELAY;
};


}
}
}
#endif
