import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

US_confirmed_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
US_deaths_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
UID_lookup_table = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv'


def seperate_by_ID(uids, us_confirmed, us_deaths):
    uid_values = []
    confirmed_separated = []
    deaths_separated = []

    for uid in range(0, len(list(uids))):
        uid_values.append(uids[uid])

        confirmed_series = us_confirmed[uid]
        confirmed_separated.append(confirmed_series)

        deaths_series = us_deaths[uid]
        deaths_separated.append(deaths_series)
    return uid_values, confirmed_separated, deaths_separated


def calculate_pct_infected(confirmed_cases, population):
    return confirmed_cases / population


def to_windowed_timeseries(uid_values, province_state_values, populations_list, confirmed_list, deaths_list, seq_len):
    x_conf = []
    x_deaths = []
    pct_infected = []
    xcounty_embedding = []
    xstate_embedding = []

    y_conf = []
    y_deaths = []

    for i in range(0, len(uid_values)):
        c_series = confirmed_list[i].values
        d_series = deaths_list[i].values

        for s in range(0, len(c_series) - seq_len):
            xcounty_embedding.append(i)
            xstate_embedding.append(province_state_values[i])

            x_conf_batch = c_series[s:s + seq_len]
            x_deaths_batch = d_series[s:s + seq_len]

            x_conf.append(x_conf_batch)
            x_deaths.append(x_deaths_batch)

            for d in range(0, seq_len):
                pct_infection_by_day = calculate_pct_infected(confirmed_cases=x_conf_batch[d],
                                                              population=populations_list[i])
                pct_infected.append(pct_infection_by_day)

            y_conf.append(c_series[s + seq_len])
            y_deaths.append(d_series[s + seq_len])

    return np.asarray(xcounty_embedding), np.asarray(xstate_embedding), np.asarray(pct_infected), \
           np.concatenate(x_conf), np.concatenate(x_deaths), np.asarray(y_conf), np.asarray(y_deaths)


def normalize_std_mean(data):
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data = (data - data_mean) / data_std
    return data, data_std, data_mean


def preprocess_data(sequence_length, buffer_size, batch_size):
    US_confirmed = pd.read_csv(US_confirmed_url)
    US_deaths = pd.read_csv(US_deaths_url)

    US_confirmed = US_confirmed.drop(columns=['iso2', 'iso3', 'code3', 'FIPS', 'Admin2',
                                              'Country_Region', 'Lat', 'Long_', 'Combined_Key'])

    US_deaths = US_deaths.drop(columns=['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State',
                                        'Country_Region', 'Lat', 'Long_', 'Combined_Key'])

    UID = US_confirmed.pop('UID')
    province_states_list = list(US_confirmed.pop('Province_State'))
    population = list(US_deaths.pop('Population').replace(to_replace=0, value=1))

    province_states_encoder = LabelEncoder()
    province_states_encoder.fit(province_states_list)
    province_states_list = province_states_encoder.transform(province_states_list)

    n_state_labels = len(list(province_states_list))
    n_county_labels = len(set(list(UID)))

    US_confirmed = US_confirmed.T
    US_deaths = US_deaths.T

    n_days = len(list(US_confirmed.index))
    print('Using data from %0d days.' % n_days)

    range_index = pd.RangeIndex(start=0, stop=n_days, step=1)

    US_confirmed.index = range_index
    US_deaths.index = range_index

    uid_values, confirmed_sep, deaths_sep = seperate_by_ID(uids=UID, us_confirmed=US_confirmed, us_deaths=US_deaths)
    xcounty_embedding, xstate_embedding, xpct_infected, x_conf, x_deaths, y_conf, y_deaths = to_windowed_timeseries(
        uid_values=uid_values,
        province_state_values=province_states_list,
        populations_list=population,
        confirmed_list=confirmed_sep,
        deaths_list=deaths_sep,
        seq_len=sequence_length)

    n_batches = len(xcounty_embedding)

    xcounty_embedding = xcounty_embedding.reshape((n_batches, 1))
    xstate_embedding = xstate_embedding.reshape((n_batches, 1))

    x_conf = x_conf.reshape((n_batches, sequence_length, 1))
    x_deaths = x_deaths.reshape((n_batches, sequence_length, 1))
    xpct_infected = xpct_infected.reshape((n_batches, sequence_length, 1))

    x_conf_norm, x_conf_std, x_conf_mean = normalize_std_mean(x_conf)
    x_deaths_norm, x_deaths_std, x_deaths_mean = normalize_std_mean(x_deaths)

    data_std = [x_conf_std, x_deaths_std]
    data_mean = [x_conf_mean, x_deaths_mean]

    x_data = np.concatenate((x_conf_norm, x_deaths_norm, xpct_infected)).reshape((n_batches, sequence_length, 3))

    y_conf = y_conf.reshape((n_batches, 1))
    y_deaths = y_deaths.reshape((n_batches, 1))

    xstate_train, xstate_test, xcounty_train, xcounty_test, x_train, x_test, y_conf_train, y_conf_test, y_deaths_train, y_deaths_test = train_test_split\
        (xstate_embedding, xcounty_embedding, x_data, y_conf, y_deaths, test_size=0.2, shuffle=True)

    train_input = (xstate_train, xcounty_train, x_train)
    test_input = (xstate_test, xcounty_test, x_test)

    train_y = (y_conf_train, y_deaths_train)
    test_y = (y_conf_test, y_deaths_test)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_y))
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).repeat()

    test_dataset = tf.data.Dataset.from_tensor_slices((test_input, test_y))
    test_dataset = test_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).repeat()

    return train_dataset, test_dataset, data_std, data_mean, n_state_labels, n_county_labels
