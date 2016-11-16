#ifndef CONSTANTS
#define CONSTANTS

#define OK                          (0)
#define NOK                         (-1)
#define ToInt(value)                static_cast<int>(value)

class MACHINE_STATE
{
public:
    enum
    {
        IDLE,
        LEARNING,
        AUTO,
    };
};

enum class UI_MENU_TREE
{
    AUTO_MAIN_MENU,
    PROGRAM_MAIN_MENU,
    PROGRAM_MANAGE_FORM,
    CONFIG_MAIN_MENU,
    UTILITY_MAIN_MENU,
    SYSTEM_MAIN_MENU,
};

enum class UI_USER_LEVEL
{
    OPERATOR,
    ENGINEER,
    SERVICE,
    DEVELOPER,
};

#endif // CONSTANTS

