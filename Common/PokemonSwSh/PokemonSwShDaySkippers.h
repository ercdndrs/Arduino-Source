/*  Day Skippers
 * 
 *  From: https://github.com/PokemonAutomation/Arduino-Source
 * 
 *      This file requires (PABB_PABOTBASE_LEVEL >= 31).
 * 
 */

#ifndef PokemonAutomation_PokemonSwShDaySkippers_H
#define PokemonAutomation_PokemonSwShDaySkippers_H

#ifdef __AVR__
#include "NativePrograms/SwitchFramework/Master.h"
#endif
#include "Common/MessageProtocol.h"
#include "Common/SwitchFramework/SwitchControllerDefs.h"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Universal
#if 0
void skipper_init_view              (void);
void skipper_auto_recovery          (void);
void skipper_rollback_year_full     (bool date_us);
void skipper_rollback_year_sync     (void);
void skipper_increment_day          (bool date_us);
void skipper_increment_month        (uint8_t days);
void skipper_increment_all          (void);
void skipper_increment_all_rollback (void);
#endif
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Client Side
#ifdef __cplusplus
namespace PokemonAutomation{
    class BotBaseContext;

    void skipper_init_view              (const BotBaseContext& context);
    void skipper_auto_recovery          (const BotBaseContext& context);
    void skipper_rollback_year_full     (const BotBaseContext& context, bool date_us);
    void skipper_rollback_year_sync     (const BotBaseContext& context);
    void skipper_increment_day          (const BotBaseContext& context, bool date_us);
    void skipper_increment_month        (const BotBaseContext& context, uint8_t days);
    void skipper_increment_all          (const BotBaseContext& context);
    void skipper_increment_all_rollback (const BotBaseContext& context);
}
#endif
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Protocols
#if _WIN32
#pragma pack(push, 1)
#define PABB_PACK
#else
#define PABB_PACK   __attribute__((packed))
#endif
////////////////////////////////////////////////////////////////////////////////

#define PABB_MSG_COMMAND_SKIPPER_INIT_VIEW                      0xbe
typedef struct{
    seqnum_t seqnum;
} PABB_PACK pabb_skipper_init_view;

#define PABB_MSG_COMMAND_SKIPPER_AUTO_RECOVERY                  0xbf
typedef struct{
    seqnum_t seqnum;
} PABB_PACK pabb_skipper_auto_recovery;

#define PABB_MSG_COMMAND_SKIPPER_ROLLBACK_YEAR_FULL             0xc0
typedef struct{
    seqnum_t seqnum;
    bool date_us;
} PABB_PACK pabb_skipper_rollback_year_full;

#define PABB_MSG_COMMAND_SKIPPER_ROLLBACK_YEAR_SYNC             0xc1
typedef struct{
    seqnum_t seqnum;
} PABB_PACK pabb_skipper_rollback_year_sync;

#define PABB_MSG_COMMAND_SKIPPER_INCREMENT_DAY                  0xc2
typedef struct{
    seqnum_t seqnum;
    bool date_us;
} PABB_PACK pabb_skipper_increment_day;

#define PABB_MSG_COMMAND_SKIPPER_INCREMENT_MONTH                0xc3
typedef struct{
    seqnum_t seqnum;
    uint8_t days;
} PABB_PACK pabb_skipper_increment_month;

#define PABB_MSG_COMMAND_SKIPPER_INCREMENT_ALL                  0xc4
typedef struct{
    seqnum_t seqnum;
} PABB_PACK pabb_skipper_increment_all;

#define PABB_MSG_COMMAND_SKIPPER_INCREMENT_ALL_ROLLBACK         0xc5
typedef struct{
    seqnum_t seqnum;
} PABB_PACK pabb_skipper_increment_all_rollback;

////////////////////////////////////////////////////////////////////////////////
#if _WIN32
#pragma pack(pop)
#endif
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#endif

