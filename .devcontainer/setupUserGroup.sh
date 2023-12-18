#!/bin/bash
UID=$1
USERNAME=$2
GID=$3
GROUPNAME=$4

echo "UID: $UID"
echo "USERNAME: $USERNAME"
echo "GID: $GID"
echo "GROUPNAME: $GROUPNAME"

delete_user_by_uid() {
    UID_TO_DELETE=$1

    if getent passwd "$UID_TO_DELETE" > /dev/null; then
        USERNAME_TO_DELETE=$(getent passwd "$UID_TO_DELETE" | cut -d: -f1)
        userdel -r "$USERNAME_TO_DELETE"
        echo "User with UID $UID_TO_DELETE has been deleted."
    else
        echo "No user with UID $UID_TO_DELETE exists."
    fi
}

create_group_by_gid_and_groupname() {
    GID=$1
    GROUPNAME=$2

    if getent group "$GID" > /dev/null; then
        echo "Group with GID $GID already exists."
    else
        groupadd -g "$GID" "$GROUPNAME"
        echo "Group $GROUPNAME with GID $GID created."
    fi
}

create_user_by_uid_and_username() {
    UID=$1
    USERNAME=$2

    if getent passwd "$UID" > /dev/null; then
        echo "User with UID $UID already exists."
    else
        useradd -u $UID -m $USERNAME -g $GID -s /bin/bash
        usermod -G sudo $USERNAME
        echo "User $USERNAME with UID $UID created."
    fi
}

delete_user_by_uid $UID
create_group_by_gid_and_groupname $GID $GROUPNAME
create_user_by_uid_and_username $UID $USERNAME
