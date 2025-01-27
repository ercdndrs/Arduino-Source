/*  Fast Code Entry
 *
 *  From: https://github.com/PokemonAutomation/Arduino-Source
 *
 */

#ifndef PokemonAutomation_PokemonSwSh_FastCodeEntry_H
#define PokemonAutomation_PokemonSwSh_FastCodeEntry_H

#include "CommonFramework/Options/FixedCode.h"
#include "NintendoSwitch/Options/TimeExpression.h"
#include "NintendoSwitch/Framework/SingleSwitchProgram.h"

namespace PokemonAutomation{
namespace NintendoSwitch{
namespace PokemonSwSh{

class FastCodeEntry : public SingleSwitchProgram{
public:
    FastCodeEntry();

    virtual void program(SingleSwitchProgramEnvironment& env) const override;

private:
    FixedCode RAID_CODE;
    TimeExpression<uint16_t> INITIAL_DELAY;
};




}
}
}
#endif



